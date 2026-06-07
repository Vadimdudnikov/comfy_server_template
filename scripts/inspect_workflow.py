#!/usr/bin/env python3
"""
Помогает собрать app/workflow.json из экспорта ComfyUI.

Один файл: узлы ComfyUI + секции _inputs, _output, _models внизу.
_inputs генерируется автоматически из всех редактируемых полей workflow.

  python scripts/inspect_workflow.py              # показать узлы
  python scripts/inspect_workflow.py --init       # дописать _inputs/_output в workflow.json
  python scripts/inspect_workflow.py --init --force   # перезаписать секции _*
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

APP_DIR = Path(__file__).resolve().parent.parent / "app"
DEFAULT_WORKFLOW = APP_DIR / "workflow.json"
META_KEYS = ("_inputs", "_output", "_models")


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


def find_output_nodes(workflow: Dict[str, Any]) -> List[str]:
    """Узлы, которые обычно отдают файл на выход (SaveImage и аналоги)."""
    output_types = {"SaveImage", "SaveAnimatedWEBP", "SaveAnimatedPNG", "SaveVideo"}
    return [
        node_id
        for node_id in sort_node_ids(list(workflow.keys()))
        if workflow[node_id].get("class_type") in output_types
    ]


def auto_build_inputs(workflow: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Собирает _inputs из всех скалярных полей всех узлов.
    Имя параметра: {node_id}_{field} — без привязки к типу pipeline.
    """
    inputs: Dict[str, Dict[str, str]] = {}

    for node_id in sort_node_ids(list(workflow.keys())):
        node = workflow[node_id]
        for field, value in node.get("inputs", {}).items():
            if not is_editable(value):
                continue
            key = f"{node_id}_{field}"
            inputs[key] = {"node": node_id, "field": field}

    return inputs


def build_meta(workflow: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    output_nodes = find_output_nodes(workflow)
    return {
        "_inputs": auto_build_inputs(workflow),
        "_output": {"node": output_nodes[0]} if output_nodes else existing.get("_output", {}),
        "_models": existing.get("_models", {}),
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
        print(f"{node_id:<6} {class_type:<28} {', '.join(fields) or '—'}")

    inputs = meta.get("_inputs") or auto_build_inputs(workflow)
    print("\n=== _inputs (параметры API) ===\n")
    print(json.dumps(inputs, indent=2, ensure_ascii=False))

    if not meta.get("_inputs"):
        print("\nЗапустите: python scripts/inspect_workflow.py --init")
    else:
        print("\nПереименуйте ключи в _inputs под свои нужды (например 45_text → prompt).")


def init_workflow(path: Path, force: bool) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    workflow, existing_meta = split_workflow(data)

    if existing_meta.get("_inputs") and not force:
        print(f"Секции _* уже есть в {path}")
        print("Используйте --force чтобы пересобрать _inputs/_output")
        sys.exit(1)

    meta = build_meta(workflow, existing_meta)
    merged = {**workflow, **meta}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Обновлён: {path}")
    print(f"  _inputs ({len(meta['_inputs'])}): {list(meta['_inputs'].keys())}")
    print(f"  _output: {meta['_output']}")
    print("Удалите лишние ключи из _inputs и переименуйте нужные.")


def main():
    parser = argparse.ArgumentParser(description="Инспектор ComfyUI workflow")
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--init", action="store_true", help="Дописать _inputs/_output в workflow.json")
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
