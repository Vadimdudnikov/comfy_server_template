#!/usr/bin/env python3
"""
comfy-kitchen >=0.2.28 использует list[int] в torch.library.custom_op,
а torch < 2.7 понимает только typing.List[int] → ComfyUI падает на импорте.

Патчим аннотации в установленном пакете, если torch слишком старый.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


def torch_supports_builtin_list_generics() -> bool:
    try:
        import torch
    except Exception:
        return False
    parts = (torch.__version__ or "0").split("+", 1)[0].split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return False
    return (major, minor) >= (2, 7)


def kitchen_root() -> Path | None:
    spec = importlib.util.find_spec("comfy_kitchen")
    if not spec or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text

    # Только аннотации параметров custom_op / type hints вида list[int|bool]
    text = re.sub(r"\blist\[int\]", "typing.List[int]", text)
    text = re.sub(r"\blist\[bool\]", "typing.List[bool]", text)

    if text == orig:
        return False

    if not re.search(r"(^|\n)import typing\b", text) and "from typing import" not in text:
        text = "import typing\n" + text

    path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    if torch_supports_builtin_list_generics():
        print("torch >= 2.7 — патч comfy_kitchen не нужен")
        return 0

    root = kitchen_root()
    if root is None:
        print("comfy_kitchen не установлен — пропускаю")
        return 0

    patched = 0
    for path in sorted(root.rglob("*.py")):
        try:
            if patch_file(path):
                print(f"patched: {path}")
                patched += 1
        except Exception as exc:
            print(f"не удалось патчить {path}: {exc}", file=sys.stderr)

    if patched:
        print(f"comfy_kitchen: исправлено файлов: {patched}")
    else:
        print("comfy_kitchen: правки не потребовались")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
