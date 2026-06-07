"""
Универсальный клиент ComfyUI API.

Один файл workflow.json:
  - узлы ComfyUI (стандартный API Format)
  - _inputs, _output, _models внизу (дописывает scripts/inspect_workflow.py --init)
"""
import copy
import io
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import websocket
from PIL import Image

COMFYUI_URL = os.getenv("COMFYUI_URL", "127.0.0.1:8188")
APP_DIR = Path(__file__).parent
WORKFLOW_PATH = Path(os.getenv("WORKFLOW_PATH", APP_DIR / "workflow.json"))


class ComfyUIClient:
    def __init__(
        self,
        server_url: str = COMFYUI_URL,
        workflow_path: Optional[Path] = None,
    ):
        self.server_url = server_url
        self.workflow_path = Path(workflow_path) if workflow_path else WORKFLOW_PATH
        self.client_id = str(uuid.uuid4())
        self.ws_url = f"ws://{server_url}/ws?clientId={self.client_id}"
        self.ws = None
        self.ws_thread = None
        self.images: Dict[str, Any] = {}
        self.execution_done = False
        self.execution_error = None
        self._workflow_template = None
        self._models_info: Dict[str, Any] = {}
        self._inputs_config: Dict[str, Any] = {}
        self._output_config: Dict[str, Any] = {}
        self._ws_initialized = False

    def _load_workflow_template(self) -> Dict[str, Any]:
        if self._workflow_template is not None:
            return self._workflow_template

        workflow_path = self.workflow_path
        try:
            with open(workflow_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"workflow.json не найден: {workflow_path}\n"
                f"Экспортируйте из ComfyUI: Save → API Format"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Ошибка парсинга {workflow_path}: {exc}") from exc

        workflow = {k: v for k, v in raw.items() if not str(k).startswith("_")}
        self._inputs_config = raw.get("_inputs", {}) or {}
        self._output_config = raw.get("_output", {}) or {}
        self._models_info = raw.get("_models", {}) or {}

        if not self._inputs_config:
            raise ValueError(
                f"Секция _inputs отсутствует в {workflow_path}\n"
                f"Запустите: python scripts/inspect_workflow.py --init"
            )

        print(f"Загружен workflow: {workflow_path} ({len(workflow)} узлов)", flush=True)
        print(f"Параметры API: {list(self._inputs_config.keys())}", flush=True)
        self._workflow_template = workflow
        return workflow

    def reload_workflow(self) -> None:
        """Сбрасывает кэш после замены workflow_template.json."""
        self._workflow_template = None
        self._models_info = {}
        self._inputs_config = {}
        self._output_config = {}
        self._load_workflow_template()

    def get_models_info(self) -> Dict[str, Any]:
        if self._workflow_template is None:
            self._load_workflow_template()
        return self._models_info

    def get_inputs_config(self) -> Dict[str, Any]:
        if self._workflow_template is None:
            self._load_workflow_template()
        return self._inputs_config

    def _apply_inputs(self, workflow: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self._inputs_config:
            raise ValueError(
                "Секция _inputs отсутствует в workflow_template.json. "
                "Добавьте маппинг параметров API → узлы ComfyUI."
            )

        for param_name, mapping in self._inputs_config.items():
            if param_name not in inputs:
                continue
            node_id = str(mapping["node"])
            field = mapping["field"]
            if node_id not in workflow:
                raise ValueError(f"Узел {node_id} не найден в workflow (параметр '{param_name}')")
            workflow[node_id]["inputs"][field] = inputs[param_name]

        return workflow

    def _queue_prompt(self, inputs: Dict[str, Any]) -> str:
        workflow = copy.deepcopy(self._load_workflow_template())
        workflow = self._apply_inputs(workflow, inputs)

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
