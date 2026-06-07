#!/usr/bin/env python3
"""Предзагрузка моделей из секции _models в workflow_template.json."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.model_downloader import ensure_models_ready, get_comfyui_base_dir


def main():
    base_dir = get_comfyui_base_dir()
    if os.getenv("COMFYUI_BASE_DIR"):
        base_dir = Path(os.getenv("COMFYUI_BASE_DIR"))

    print(f"ComfyUI base dir: {base_dir}")
    success = ensure_models_ready(base_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
