#!/usr/bin/env python3
"""
Помогает собрать app/workflow.json из экспорта ComfyUI.

Один файл: узлы ComfyUI + секции _inputs, _output, _models внизу.
Секции _* собираются автоматически из структуры workflow — без привязки к конкретному pipeline.

  python scripts/inspect_workflow.py              # показать узлы и метаданные
  python scripts/inspect_workflow.py --init       # дописать _inputs/_output/_models
  python scripts/inspect_workflow.py --init --force   # пересобрать секции _*
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

APP_DIR = Path(__file__).resolve().parent.parent / "app"
DEFAULT_WORKFLOW = APP_DIR / "workflow.json"
META_KEYS = ("_inputs", "_output", "_models")

# Узлы-загрузчики: class_type → (категория _models, поле с именем файла, папка в models/)
LOADER_NODES: Dict[str, Tuple[str, str, str]] = {
    "CLIPLoader": ("clip", "clip_name", "clip"),
    "VAELoader": ("vae", "vae_name", "vae"),
    "UNETLoader": ("unet", "unet_name", "diffusion_models"),
    "CheckpointLoaderSimple": ("checkpoint", "ckpt_name", "checkpoints"),
    "LoraLoader": ("lora", "lora_name", "loras"),
    "ControlNetLoader": ("controlnet", "control_net_name", "controlnet"),
}

# Узлы, которые отдают файл на выход
OUTPUT_CLASS_TYPES = frozenset({
    "SaveImage",
    "SaveAnimatedWEBP",
    "SaveAnimatedPNG",
    "SaveVideo",
})

# Поля, которые не нужны в API (конфиг загрузчиков, префиксы файлов и т.п.)
SKIP_INPUT_FIELDS = frozenset({
    "clip_name",
    "vae_name",
    "unet_name",
    "ckpt_name",
    "lora_name",
    "control_net_name",
    "type",
    "device",
    "weight_dtype",
    "filename_prefix",
})


def is_editable(value: Any) -> bool:
    """Скалярные значения — не ссылки на другие узлы."""
    return isinstance(value, (str, int, float, bool))


def sort_node_ids(node_ids: List[str]) -> List[str]:
    def key(node_id: str):
        return (0, int(node_id)) if node_id.isdigit() else (1, node_id)

    return sorted(node_ids, key=key)


def split_workflow(data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    workflow = {k: v for k, v in data.items() if not str(k).startswith("_")}
    meta = {k: data[k] for k in META_KEYS if k in data}
    return workflow, meta


def is_loader_node(node: Dict[str, Any]) -> bool:
    return node.get("class_type") in LOADER_NODES


def find_output_nodes(workflow: Dict[str, Any]) -> List[str]:
    return [
        node_id
        for node_id in sort_node_ids(list(workflow.keys()))
        if workflow[node_id].get("class_type") in OUTPUT_CLASS_TYPES
    ]


def auto_build_inputs(workflow: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Собирает _inputs из скалярных полей узлов.
    Пропускает loader-узлы и служебные поля — модели задаются в _models.
    Имя параметра: {node_id}_{field}.
    """
    inputs: Dict[str, Dict[str, str]] = {}

    for node_id in sort_node_ids(list(workflow.keys())):
        node = workflow[node_id]
        if is_loader_node(node):
            continue

        for field, value in node.get("inputs", {}).items():
            if field in SKIP_INPUT_FIELDS or not is_editable(value):
                continue
            key = f"{node_id}_{field}"
            inputs[key] = {"node": node_id, "field": field}

    return inputs


def auto_build_models(
    workflow: Dict[str, Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Дополняет _models по loader-узлам workflow.
    Уже заполненные записи (url, path, description) не перезаписываются.
    """
    models = copy.deepcopy(existing) if existing else {}

    for node_id in sort_node_ids(list(workflow.keys())):
        node = workflow[node_id]
        class_type = node.get("class_type")
        loader_info = LOADER_NODES.get(class_type)
        if not loader_info:
            continue

        category, field, default_dir = loader_info
        filename = node.get("inputs", {}).get(field)
        if not isinstance(filename, str) or not filename:
            continue

        models.setdefault(category, {})
        if filename in models[category]:
            continue

        models[category][filename] = {
            "url": "",
            "path": f"models/{default_dir}/{filename}",
            "description": f"{class_type} (узел {node_id})",
        }

    return models


def build_meta(workflow: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    output_nodes = find_output_nodes(workflow)
    return {
        "_inputs": auto_build_inputs(workflow),
        "_output": {"node": output_nodes[0]} if output_nodes else existing.get("_output", {}),
        "_models": auto_build_models(workflow, existing.get("_models", {})),
    }


def print_workflow_table(workflow: Dict[str, Any], meta: Dict[str, Any]) -> None:
    print("\n=== Узлы workflow ===\n")
    print(f"{'ID':<6} {'class_type':<28} {'редактируемые поля'}")
    print("-" * 72)

    for node_id in sort_node_ids(list(workflow.keys())):
        node = workflow[node_id]
        class_type = node.get("class_type", "?")
        fields = [
            f"{k}={json.dumps(v, ensure_ascii=False)}"
            for k, v in node.get("inputs", {}).items()
            if is_editable(v)
        ]
        marker = " [loader]" if is_loader_node(node) else ""
        print(f"{node_id:<6} {class_type:<28}{marker} {', '.join(fields) or '—'}")

    inputs = meta.get("_inputs") or auto_build_inputs(workflow)
    print("\n=== _inputs (параметры API) ===\n")
    print(json.dumps(inputs, indent=2, ensure_ascii=False))

    output_nodes = find_output_nodes(workflow)
    output = meta.get("_output") or ({"node": output_nodes[0]} if output_nodes else {})
    print("\n=== _output ===\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))

    models = meta.get("_models") or auto_build_models(workflow, {})
    print("\n=== _models ===\n")
    if not models:
        print("(пусто — loader-узлы не найдены или секция не заполнена)")
    else:
        for category, entries in models.items():
            print(f"  [{category}]")
            for name, info in entries.items():
                url = info.get("url", "")
                status = "ok" if url else "нет url"
                print(f"    {name}: {info.get('path', '?')} ({status})")

    if not meta.get("_inputs"):
        print("\nЗапустите: python scripts/inspect_workflow.py --init")
    else:
        print("\nУдалите лишние ключи в _inputs и переименуйте нужные (например 45_text → prompt).")
        missing_urls = [
            f"{cat}/{name}"
            for cat, entries in models.items()
            for name, info in entries.items()
            if not info.get("url")
        ]
        if missing_urls:
            print(f"Допишите url в _models: {', '.join(missing_urls)}")


def init_workflow(path: Path, force: bool) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    workflow, existing_meta = split_workflow(data)

    if existing_meta.get("_inputs") and not force:
        print(f"Секции _* уже есть в {path}")
        print("Используйте --force чтобы пересобрать _inputs/_output/_models")
        sys.exit(1)

    meta = build_meta(workflow, existing_meta)
    merged = {**workflow, **meta}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Обновлён: {path}")
    print(f"  _inputs ({len(meta['_inputs'])}): {list(meta['_inputs'].keys())}")
    print(f"  _output: {meta['_output']}")
    model_count = sum(len(v) for v in meta["_models"].values())
    print(f"  _models ({model_count} файлов)")
    missing_urls = [
        f"{cat}/{name}"
        for cat, entries in meta["_models"].items()
        for name, info in entries.items()
        if not info.get("url")
    ]
    if missing_urls:
        print(f"  Допишите url: {', '.join(missing_urls)}")
    print("Удалите лишние ключи из _inputs и переименуйте нужные.")


def main():
    parser = argparse.ArgumentParser(description="Инспектор ComfyUI workflow")
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--init", action="store_true", help="Дописать _inputs/_output/_models в workflow.json")
    parser.add_argument("--force", action="store_true", help="Пересобрать секции _*")
    args = parser.parse_args()

    if not args.workflow.exists():
        print(f"Не найден: {args.workflow}")
        print("Вставьте экспорт ComfyUI (Save → API Format) в app/workflow.json")
        sys.exit(1)

    if args.init:
        init_workflow(args.workflow, args.force)
        return

    with open(args.workflow, "r", encoding="utf-8") as f:
        data = json.load(f)
    workflow, meta = split_workflow(data)

    print(f"Workflow: {args.workflow} ({len(workflow)} узлов)")
    print_workflow_table(workflow, meta)


if __name__ == "__main__":
    main()
