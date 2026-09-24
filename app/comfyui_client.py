"""
Универсальный клиент ComfyUI API.

Workflow читается из папки workflow/ (сырой UI-экспорт ComfyUI).
Конвертация и _inputs/_output/_models — автоматически при загрузке.
"""
import base64
import copy
import io
import json
import mimetypes
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import websocket
from PIL import Image

from .workflow_loader import (
    LoadedWorkflow,
    build_meta,
    load_workflow,
    max_reference_images,
    resolve_workflow_source,
    select_image_edit_path,
    prepare_workflow_data,
)

COMFYUI_URL = os.getenv("COMFYUI_URL", "127.0.0.1:8188")
DATA_URL_RE = re.compile(r"^data:([^;]+);base64,(.+)$", re.DOTALL)



class ComfyUIClient:
    def __init__(
        self,
        server_url: str = COMFYUI_URL,
        workflow_path: Optional[Path] = None,
    ):
        self.server_url = server_url
        self.workflow_path = Path(workflow_path) if workflow_path else resolve_workflow_source()
        self.client_id = str(uuid.uuid4())
        self.ws_url = f"ws://{server_url}/ws?clientId={self.client_id}"
        self.ws = None
        self.ws_thread = None
        self.images: Dict[str, Any] = {}
        self.execution_done = False
        self.execution_error = None
        self._workflow_template = None
        self._loaded: Optional[LoadedWorkflow] = None
        self._models_info: Dict[str, Any] = {}
        self._inputs_config: Dict[str, Any] = {}
        self._output_config: Dict[str, Any] = {}
        self._raw_ui: Optional[Dict[str, Any]] = None
        self._variant_cache: Dict[int, Dict[str, Any]] = {}
        self._ws_initialized = False

    def _load_workflow_template(self) -> Dict[str, Any]:
        if self._workflow_template is not None:
            return self._workflow_template

        loaded = load_workflow(self.workflow_path)
        self._loaded = loaded
        self._workflow_template = loaded.workflow
        self._inputs_config = loaded.inputs_config
        self._output_config = loaded.output_config
        self._models_info = loaded.models_info
        self._raw_ui = loaded.raw_data
        self._variant_cache = {}

        print(
            f"Загружен workflow: {loaded.source_path} "
            f"({loaded.source_format}, {len(loaded.workflow)} узлов)",
            flush=True,
        )
        print(f"Параметры API: {list(self._inputs_config.keys())}", flush=True)
        return self._workflow_template

    def reload_workflow(self) -> None:
        """Сбрасывает кэш после замены файла в workflow/."""
        self._workflow_template = None
        self._loaded = None
        self._models_info = {}
        self._inputs_config = {}
        self._output_config = {}
        self._raw_ui = None
        self._variant_cache = {}
        self.workflow_path = resolve_workflow_source(self.workflow_path)
        self._load_workflow_template()

    def get_models_info(self) -> Dict[str, Any]:
        if self._workflow_template is None:
            self._load_workflow_template()
        return self._models_info

    def get_inputs_config(self) -> Dict[str, Any]:
        if self._workflow_template is None:
            self._load_workflow_template()
        return self._inputs_config

    def _looks_like_image_value(self, value: Any) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        text = value.strip()
        if text.startswith("data:image/"):
            return True
        if text.startswith("http://") or text.startswith("https://"):
            return True
        # Сырой base64 (длинная строка без расширения файла)
        if len(text) > 256 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", text):
            return True
        return False

    def _download_url(self, url: str) -> Tuple[bytes, str]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = response.read()
                content_type = response.headers.get("Content-Type", "image/png")
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Не удалось скачать изображение ({e.code} {e.reason}): {url}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Не удалось скачать изображение: {url} ({e})") from e

        if not raw:
            raise RuntimeError(f"Пустой ответ при скачивании изображения: {url}")

        mime = content_type.split(";")[0].strip() or "image/png"
        ext = mimetypes.guess_extension(mime) or Path(urllib.parse.urlparse(url).path).suffix or ".png"
        if ext == ".jpe":
            ext = ".jpg"
        name = Path(urllib.parse.urlparse(url).path).name or f"upload{ext}"
        if "." not in name:
            name = f"{name}{ext}"
        return raw, name

    def _decode_image_payload(self, value: str) -> Tuple[bytes, str]:
        text = value.strip()
        match = DATA_URL_RE.match(text)
        if match:
            mime = match.group(1).strip()
            raw = base64.b64decode(match.group(2))
            ext = mimetypes.guess_extension(mime) or ".png"
            if ext == ".jpe":
                ext = ".jpg"
            return raw, f"upload{ext}"

        if text.startswith("http://") or text.startswith("https://"):
            return self._download_url(text)

        # Сырой base64
        raw = base64.b64decode(text)
        return raw, "upload.png"

    def _upload_image_bytes(self, image_bytes: bytes, filename: str) -> str:
        boundary = f"----ComfyUpload{uuid.uuid4().hex}"
        filename = Path(filename).name or "upload.png"
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        body = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode(),
                image_bytes,
                b"\r\n",
                f"--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="type"\r\n\r\n',
                b"input\r\n",
                f"--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="overwrite"\r\n\r\n',
                b"true\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )

        req = urllib.request.Request(
            f"http://{self.server_url}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else "No error body"
            raise RuntimeError(f"ComfyUI upload failed ({e.code}): {error_body[:500]}") from e

        name = result.get("name")
        if not name:
            raise RuntimeError(f"ComfyUI upload вернул неожиданный ответ: {result}")
        subfolder = result.get("subfolder") or ""
        return f"{subfolder}/{name}" if subfolder else name

    def _ensure_comfy_image(self, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Параметр изображения должен быть строкой (filename, URL или base64)")
        text = value.strip()
        if not text:
            raise ValueError("Пустое значение изображения")
        if not self._looks_like_image_value(text):
            # Уже имя файла в input/ ComfyUI
            return text

        raw, filename = self._decode_image_payload(text)
        uploaded = self._upload_image_bytes(raw, filename)
        print(f"Загружен референс в ComfyUI: {uploaded}", flush=True)
        return uploaded

    def _count_request_images(self, inputs: Dict[str, Any]) -> int:
        if "images" in inputs and isinstance(inputs["images"], (list, tuple)):
            return len(inputs["images"])
        if isinstance(inputs.get("image"), (list, tuple)):
            return len(inputs["image"])
        count = 0
        if isinstance(inputs.get("image"), str) and inputs.get("image", "").strip():
            count += 1
        index = 1
        while True:
            key = f"image_{index}"
            if key not in inputs:
                break
            if isinstance(inputs[key], str) and str(inputs[key]).strip():
                count += 1
            index += 1
        return count

    def _workflow_for_images(self, num_images: int) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Возвращает (workflow, inputs_config, output_config) под N картинок."""
        self._load_workflow_template()
        if not self._raw_ui:
            return (
                self._workflow_template,
                self._inputs_config,
                self._output_config,
            )

        key = 2 if num_images >= 2 else 1
        if key not in self._variant_cache:
            selected = select_image_edit_path(self._raw_ui, key)
            workflow, meta, _ = prepare_workflow_data(selected)
            built = build_meta(workflow, meta)
            max_images = max_reference_images(self._raw_ui)
            if max_images > 0:
                built["_inputs"]["images"] = {
                    "type": "image_array",
                    "field": "image",
                    "nodes": [
                        nid
                        for nid, node in workflow.items()
                        if node.get("class_type") == "LoadImage"
                    ],
                    "max_items": max_images,
                    "min_items": 1,
                }
            self._variant_cache[key] = {
                "workflow": workflow,
                "inputs_config": built["_inputs"],
                "output_config": built["_output"],
            }
            print(
                f"Вариант workflow для {key} image(s): "
                f"{len(workflow)} узлов, inputs={list(built['_inputs'].keys())}",
                flush=True,
            )

        variant = self._variant_cache[key]
        return variant["workflow"], variant["inputs_config"], variant["output_config"]

    def _image_alias_params(self, inputs_config: Optional[Dict[str, Any]] = None) -> list[str]:
        """Порядок слотов референсов: image, image_1, image_2, ..."""
        cfg = inputs_config if inputs_config is not None else self._inputs_config
        params: list[str] = []
        if "image" in cfg and cfg["image"].get("type") != "image_array":
            params.append("image")
        index = 1
        while f"image_{index}" in cfg:
            params.append(f"image_{index}")
            index += 1
        return params

    def _expand_images_array(
        self,
        inputs: Dict[str, Any],
        inputs_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prepared = dict(inputs)
        image_params = self._image_alias_params(inputs_config)

        images_list = None
        if "images" in prepared:
            images_list = prepared.pop("images")
        elif isinstance(prepared.get("image"), (list, tuple)):
            images_list = prepared.pop("image")

        if images_list is None:
            return prepared

        if not isinstance(images_list, (list, tuple)):
            raise ValueError("images должен быть массивом строк (base64 / data URL / URL / filename)")
        if not image_params:
            raise ValueError("В текущем workflow нет LoadImage для референсов")
        if len(images_list) > len(image_params):
            raise ValueError(
                f"Передано {len(images_list)} картинок, а в workflow только {len(image_params)} "
                f"слот(ов): {', '.join(image_params)}"
            )
        if len(images_list) < 1:
            raise ValueError("images не должен быть пустым")
        for param_name, value in zip(image_params, images_list):
            prepared[param_name] = value
        return prepared

    def _prepare_inputs(
        self,
        inputs: Dict[str, Any],
        inputs_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._workflow_template is None:
            self._load_workflow_template()

        cfg = inputs_config if inputs_config is not None else self._inputs_config
        prepared = self._expand_images_array(inputs, cfg)

        image_params = self._image_alias_params(cfg)
        if image_params:
            provided = [p for p in image_params if p in prepared and str(prepared[p]).strip()]
            if not provided:
                max_items = (cfg.get("images") or {}).get("max_items", len(image_params))
                raise ValueError(
                    "Нужен хотя бы один референс: "
                    f'"images": ["https://..."] (до {max_items})'
                )

        for param_name, mapping in cfg.items():
            if param_name not in prepared:
                continue
            if mapping.get("type") == "image_array":
                continue
            field = str(mapping.get("field", ""))
            if field == "image" or field.startswith("image_"):
                prepared[param_name] = self._ensure_comfy_image(prepared[param_name])
        return prepared

    def _apply_inputs(
        self,
        workflow: Dict[str, Any],
        inputs: Dict[str, Any],
        inputs_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = inputs_config if inputs_config is not None else self._inputs_config
        if not cfg:
            raise ValueError(
                "Не удалось собрать параметры API из workflow/. "
                "Проверьте файл в папке workflow/."
            )

        prepared = self._prepare_inputs(inputs, cfg)

        for param_name, mapping in cfg.items():
            if param_name not in prepared:
                continue
            if mapping.get("type") == "image_array":
                continue
            if "node" not in mapping:
                continue
            node_id = str(mapping["node"])
            field = mapping["field"]
            if node_id not in workflow:
                raise ValueError(f"Узел {node_id} не найден в workflow (параметр '{param_name}')")
            workflow[node_id]["inputs"][field] = prepared[param_name]

        return workflow

    def _queue_prompt(self, inputs: Dict[str, Any]) -> str:
        num_images = self._count_request_images(inputs)
        workflow_template, inputs_config, output_config = self._workflow_for_images(num_images)
        self._output_config = output_config

        workflow = copy.deepcopy(workflow_template)
        workflow = self._apply_inputs(workflow, inputs, inputs_config)

        payload = {"prompt": workflow, "client_id": self.client_id}
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            f"http://{self.server_url}/prompt",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            response = urllib.request.urlopen(req)
            result = json.loads(response.read())
            print(f"ComfyUI принял запрос, prompt_id: {result.get('prompt_id', 'unknown')}", flush=True)
            return result["prompt_id"]
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else "No error body"
            print(f"ComfyUI HTTP Error {e.code}: {error_body[:500]}", flush=True)
            raise RuntimeError(f"ComfyUI HTTP Error {e.code}: {error_body[:500]}") from e

    def refresh_models(self) -> bool:
        try:
            url = f"http://{self.server_url}/object_info"
            req = urllib.request.Request(url)
            urllib.request.urlopen(req, timeout=10)
            print("Список моделей в ComfyUI обновлён", flush=True)
            return True
        except Exception as e:
            print(f"Не удалось обновить список моделей: {e}", flush=True)
            return False

    def _get_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url = f"http://{self.server_url}/view?{urllib.parse.urlencode(data)}"
        with urllib.request.urlopen(url) as response:
            return response.read()

    def _get_history(self, prompt_id: str) -> Optional[Dict]:
        try:
            url = f"http://{self.server_url}/history/{prompt_id}"
            with urllib.request.urlopen(url, timeout=10) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def _on_message(self, ws: websocket.WebSocketApp, message: str):
        try:
            if not isinstance(message, str):
                return
            message = json.loads(message)
            msg_type = message.get("type", "")

            if msg_type == "executing":
                node = message.get("data", {}).get("node")
                if node is None:
                    self.execution_done = True
            elif msg_type == "executed":
                output = message.get("data", {}).get("output", {})
                node_id = message.get("data", {}).get("node")
                if "images" in output:
                    self.images[node_id] = output["images"]
            elif msg_type == "execution_error":
                error_data = message.get("data", {})
                self.execution_error = {
                    "type": error_data.get("exception_type", "Unknown"),
                    "node_id": error_data.get("node_id", "Unknown"),
                    "node_type": error_data.get("node_type", "Unknown"),
                    "message": error_data.get("exception_message", ""),
                }
                self.execution_done = True
        except Exception as e:
            print(f"Ошибка WebSocket: {e}", flush=True)

    def _ensure_websocket(self) -> None:
        if self._ws_initialized and self.ws is not None and self.ws_thread and self.ws_thread.is_alive():
            return

        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self._on_message,
            on_error=lambda ws, err: print(f"WebSocket error: {err}", flush=True),
            on_close=lambda ws, code, msg: None,
            on_open=lambda ws: None,
        )
        self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.ws_thread.start()
        time.sleep(1.5)
        self._ws_initialized = True

    def _find_output_node(self, outputs: Dict[str, Any]) -> Optional[str]:
        preferred = self._output_config.get("node")
        if preferred and str(preferred) in outputs and "images" in outputs[str(preferred)]:
            return str(preferred)

        for node_id, node_output in outputs.items():
            if "images" in node_output:
                return node_id
        return None

    def run_workflow(
        self,
        inputs: Dict[str, Any],
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """
        Запускает workflow с переданными inputs.
        Возвращает dict с PIL Image и метаданными ComfyUI.
        """
        self.execution_done = False
        self.images = {}
        self.execution_error = None

        self._ensure_websocket()
        prompt_id = self._queue_prompt(inputs)

        start_time = time.time()
        while not self.execution_done:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"ComfyUI не завершил выполнение за {timeout}с")
            time.sleep(0.2)

        if self.execution_error:
            err = self.execution_error
            raise RuntimeError(
                f"ComfyUI error in node {err['node_id']} ({err['node_type']}): "
                f"{err['type']} - {err['message']}"
            )

        time.sleep(2)

        history = None
        for retry in range(5):
            history = self._get_history(prompt_id)
            if history and prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI execution failed: {status.get('messages')}")
                break
            if retry < 4:
                time.sleep(2)

        if not history or prompt_id not in history:
            raise RuntimeError(f"Не удалось получить историю для prompt_id {prompt_id}")

        outputs = history[prompt_id].get("outputs", {})
        image_node_id = self._find_output_node(outputs)
        if not image_node_id:
            raise RuntimeError("В выводе ComfyUI не найдено изображение. Проверьте _output в JSON.")

        images = outputs[image_node_id]["images"]
        if not images:
            raise RuntimeError("Список изображений пуст")

        image_info = images[0]
        image_data = self._get_image(
            image_info["filename"],
            image_info.get("subfolder", ""),
            image_info.get("type", "output"),
        )
        image = Image.open(io.BytesIO(image_data))

        return {
            "image": image,
            "prompt_id": prompt_id,
            "output_node": image_node_id,
            "filename": image_info["filename"],
        }
