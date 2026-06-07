"""
Автозагрузка моделей из секции _models в workflow_template.json.
"""
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from .comfyui_client import ComfyUIClient, WORKFLOW_PATH


def _is_safetensors_corrupted(path: Path) -> bool:
    if path.suffix.lower() != ".safetensors":
        return False
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            _ = list(f.keys())
        return False
    except Exception as e:
        print(f"Файл повреждён ({path.name}): {e}", flush=True)
        return True


def get_comfyui_base_dir() -> Path:
    base_dir = os.getenv("COMFYUI_BASE_DIR")
    if base_dir:
        return Path(base_dir)
    return Path("/home/ComfyUI")


def download_model(url: str, destination: Path, description: str = "") -> bool:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and destination.stat().st_size > 0:
            print(f"Модель уже есть: {destination.name}", flush=True)
            return True

        print(f"Скачиваю: {description or destination.name}", flush=True)
        print(f"  URL: {url}", flush=True)

        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) / total_size)
                print(f"\r  Прогресс: {percent:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(url, destination, reporthook=show_progress)
        print(flush=True)

        if not destination.exists() or destination.stat().st_size == 0:
            return False

        if destination.suffix.lower() == ".safetensors" and _is_safetensors_corrupted(destination):
            destination.unlink(missing_ok=True)
            return False

        size_mb = destination.stat().st_size / 1024 / 1024
        print(f"Скачано: {destination.name} ({size_mb:.2f} MB)", flush=True)
        return True
    except Exception as e:
        print(f"Ошибка скачивания {destination.name}: {e}", flush=True)
        if destination.exists():
            destination.unlink(missing_ok=True)
        return False


def ensure_models_ready(
    comfyui_base_dir: Optional[Path] = None,
    workflow_path: Optional[Path] = None,
) -> bool:
    if comfyui_base_dir is None:
        comfyui_base_dir = get_comfyui_base_dir()

    client = ComfyUIClient(workflow_path=workflow_path or WORKFLOW_PATH)
    models_info = client.get_models_info()

    if not models_info:
        print("Секция _models не найдена — пропускаю загрузку моделей", flush=True)
        return True

    print(f"Проверяю модели (ComfyUI: {comfyui_base_dir})", flush=True)

    models_to_check: list[Dict[str, Any]] = []
    for model_type, models_dict in models_info.items():
        for model_name, model_info in models_dict.items():
            url = model_info.get("url", "")
            relative_path = model_info.get("path", "")
            if not url:
                print(f"Пропуск {model_name}: URL не указан", flush=True)
                continue

            if relative_path.startswith("models/"):
                model_path = relative_path.replace("models/", "", 1)
                destination = comfyui_base_dir / "models" / model_path
            else:
                destination = comfyui_base_dir / relative_path

            models_to_check.append({
                "name": model_name,
                "url": url,
                "path": destination,
                "description": model_info.get("description", model_name),
                "type": model_type,
            })

    if not models_to_check:
        return True

    all_success = True
    new_models_downloaded = False

    for model in models_to_check:
        path = model["path"]
        need_download = (
            not path.exists()
            or path.stat().st_size == 0
            or (path.suffix.lower() == ".safetensors" and _is_safetensors_corrupted(path))
        )
        if need_download:
            if path.exists():
                path.unlink(missing_ok=True)
            success = download_model(model["url"], path, f"{model['type']}/{model['name']}")
            new_models_downloaded = new_models_downloaded or success
            all_success = all_success and success
        else:
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"Готово: {model['name']} ({size_mb:.2f} MB)", flush=True)

    if all_success and new_models_downloaded:
        import time
        time.sleep(1)
        client.refresh_models()

    return all_success
