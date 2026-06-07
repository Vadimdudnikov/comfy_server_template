#!/usr/bin/env python3
"""
Опциональный инспектор workflow (для отладки).

Сервер сам читает сырой UI-экспорт из workflow/ — скрипт не обязателен.

  python scripts/inspect_workflow.py
  python scripts/inspect_workflow.py --workflow workflow/workflow.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.workflow_loader import (
    WORKFLOW_DIR,
    auto_build_inputs,
    auto_build_models,
    build_meta,
    find_output_nodes,
    has_subgraph_ids,
    is_editable,
    is_loader_node,
    is_ui_workflow,
    load_workflow,
    merge_models,
    prepare_workflow_data,
    resolve_workflow_source,
    sort_node_ids,
    split_workflow,
)

DEFAULT_WORKFLOW = WORKFLOW_DIR / "workflow.json"


def print_workflow_table(workflow, meta) -> None:
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
        print(f"{node_id:<6} {class_type:<28}{marker} {', '.join(fields) or '-'}")

    inputs = meta.get("_inputs") or auto_build_inputs(workflow)
    print("\n=== _inputs (параметры API) ===\n")
    print(json.dumps(inputs, indent=2, ensure_ascii=False))

    output_nodes = find_output_nodes(workflow)
    output = meta.get("_output") or ({"node": output_nodes[0]} if output_nodes else {})
    print("\n=== _output ===\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))

    models = meta.get("_models") or merge_models(
        auto_build_models(workflow, {}),
        meta.get("_models_imported", {}),
    )
    print("\n=== _models ===\n")
    for category, entries in models.items():
        print(f"  [{category}]")
        for name, info in entries.items():
            url = info.get("url", "")
            status = "ok" if url else "нет url"
            print(f"    {name}: {info.get('path', '?')} ({status})")


def main():
    parser = argparse.ArgumentParser(description="Инспектор ComfyUI workflow")
    parser.add_argument("--workflow", type=Path, default=None, help="Файл workflow для просмотра")
    args = parser.parse_args()

    path = args.workflow or resolve_workflow_source()
    if not path.exists():
        print(f"Не найден: {path}")
        print("Положите UI-экспорт ComfyUI в workflow/workflow.json")
        sys.exit(1)

    loaded = load_workflow(path)
    print(f"Workflow: {loaded.source_path} ({loaded.source_format}, {len(loaded.workflow)} узлов)")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if is_ui_workflow(raw):
        print("(конвертировано из UI Format)")
    elif has_subgraph_ids(split_workflow(raw)[0]):
        print("(subgraph id нормализованы: 76:8 -> 8)")

    meta = {
        "_inputs": loaded.inputs_config,
        "_output": loaded.output_config,
        "_models": loaded.models_info,
    }
    print_workflow_table(loaded.workflow, meta)


if __name__ == "__main__":
    main()
