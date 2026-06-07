import os
from pathlib import Path
from typing import Any, Dict, Optional

from .comfyui_client import ComfyUIClient

_client: Optional[ComfyUIClient] = None


def get_client() -> ComfyUIClient:
    global _client
    if _client is None:
        comfyui_url = os.getenv("COMFYUI_URL", "127.0.0.1:8188")
        _client = ComfyUIClient(server_url=comfyui_url)
    return _client


def reload_client() -> None:
    global _client
    if _client:
        _client.reload_workflow()
    else:
        get_client().reload_workflow()


def run_pipeline(inputs: Dict[str, Any], filename: str = "output.png") -> str:
    """
    Запускает pipeline и сохраняет результат в static/.
    Возвращает путь к файлу.
    """
    client = get_client()
    result = client.run_workflow(inputs)

    static_dir = Path(os.getenv("STATIC_DIR", "static"))
    out_path = static_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result["image"].save(out_path)
    return str(out_path)
