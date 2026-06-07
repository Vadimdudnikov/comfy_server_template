"""
Загрузка и конвертация workflow из папки workflow/.

Поддерживает сырой UI-экспорт ComfyUI (Save workflow), API Format и API с subgraph.
При старте сервис сам строит _inputs, _output, _models — скрипты не нужны.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = Path(os.getenv("WORKFLOW_DIR", BASE_DIR / "workflow"))

META_KEYS = ("_inputs", "_output", "_models")

LOADER_NODES: Dict[str, Tuple[str, str, str]] = {
    "CLIPLoader": ("clip", "clip_name", "clip"),
    "VAELoader": ("vae", "vae_name", "vae"),
    "UNETLoader": ("unet", "unet_name", "diffusion_models"),
    "CheckpointLoaderSimple": ("checkpoint", "ckpt_name", "checkpoints"),
    "LoraLoader": ("lora", "lora_name", "loras"),
    "LoraLoaderModelOnly": ("lora", "lora_name", "loras"),
    "ControlNetLoader": ("controlnet", "control_net_name", "controlnet"),
}

SKIP_CLASS_TYPES = frozenset({
    "PreviewAny",
    "Note",
    "MarkdownNote",
})

CLIP_TYPE_DIRS: Dict[str, str] = {
    "qwen_image": "text_encoders",
    "lumina2": "clip",
}

OUTPUT_CLASS_TYPES = frozenset({
    "SaveImage",
    "SaveAnimatedWEBP",
    "SaveAnimatedPNG",
    "SaveVideo",
})

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
    "string_b",
    "delimiter",
})

EXTRA_WIDGET_FIELDS: Dict[str, List[str]] = {
    "CLIPLoader": ["type", "device"],
    "UNETLoader": ["weight_dtype"],
    "KSampler": ["seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "CLIPTextEncode": ["text"],
    "SaveImage": ["filename_prefix"],
    "ModelSamplingAuraFlow": ["shift"],
    "StringConcatenate": ["string_b", "delimiter"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    "LoraLoaderModelOnly": ["lora_name", "strength_model"],
}

MODEL_DIR_TO_CATEGORY: Dict[str, str] = {
    "vae": "vae",
    "clip": "clip",
    "text_encoders": "clip",
    "diffusion_models": "unet",
    "loras": "lora",
    "checkpoints": "checkpoint",
    "controlnet": "controlnet",
}


@dataclass
class LoadedWorkflow:
    source_path: Path
    workflow: Dict[str, Any]
    inputs_config: Dict[str, Any]
    output_config: Dict[str, Any]
    models_info: Dict[str, Any]
    source_format: str


def normalize_workflow_name(name: str) -> str:
    name = name.strip()
    if name.lower().endswith(".json"):
        return name[:-5]
    return name


def resolve_workflow_by_name(workflow_dir: Path, name: str) -> Path:
    normalized = normalize_workflow_name(name)
    if not normalized:
        raise FileNotFoundError("WORKFLOW_NAME пустой")

    candidate = workflow_dir / f"{normalized}.json"
    if candidate.exists():
        return candidate

    available = sorted(p.stem for p in workflow_dir.glob("*.json"))
    hint = f"Доступные: {', '.join(available)}" if available else "Папка workflow/ пуста"
    raise FileNotFoundError(
        f"Workflow '{normalized}' не найден в {workflow_dir}\n{hint}"
    )


def resolve_workflow_source(path: Optional[Path] = None) -> Path:
    if path is not None:
        return Path(path)

    env_path = os.getenv("WORKFLOW_PATH")
    if env_path:
        return Path(env_path)

    workflow_dir = Path(os.getenv("WORKFLOW_DIR", WORKFLOW_DIR))

    workflow_name = os.getenv("WORKFLOW_NAME", "").strip()
    if not workflow_name:
        available = sorted(p.stem for p in workflow_dir.glob("*.json"))
        hint = f"Доступные: {', '.join(available)}" if available else "Папка workflow/ пуста"
        raise FileNotFoundError(
            "WORKFLOW_NAME не задан. Без него запускается только ComfyUI.\n"
            f"Для API-сервиса: WORKFLOW_NAME=<name> (без .json). {hint}"
        )

    return resolve_workflow_by_name(workflow_dir, workflow_name)


def is_editable(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def sort_node_ids(node_ids: List[str]) -> List[str]:
    def key(node_id: str):
        return (0, int(node_id)) if node_id.isdigit() else (1, node_id)

    return sorted(node_ids, key=key)


def split_workflow(data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    workflow = {k: v for k, v in data.items() if not str(k).startswith("_")}
    meta = {k: data[k] for k in META_KEYS if k in data}
    return workflow, meta


def is_ui_workflow(data: Dict[str, Any]) -> bool:
    return isinstance(data.get("nodes"), list)


def is_api_workflow(data: Dict[str, Any]) -> bool:
    for key, value in data.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, dict) and "class_type" in value:
            return True
    return False


def has_subgraph_ids(workflow: Dict[str, Any]) -> bool:
    return any(":" in str(node_id) for node_id in workflow)


def normalize_node_id(node_id: str) -> str:
    return node_id.split(":", 1)[1] if ":" in node_id else node_id


def remap_input_value(value: Any, id_map: Dict[str, str]) -> Any:
    if (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    ):
        ref = id_map.get(value[0], normalize_node_id(value[0]))
        return [ref, value[1]]
    return value


def normalize_api_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    id_map = {node_id: normalize_node_id(str(node_id)) for node_id in workflow}
    normalized: Dict[str, Any] = {}

    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        if class_type in SKIP_CLASS_TYPES:
            continue

        new_id = id_map[str(node_id)]
        inputs = {
            field: remap_input_value(val, id_map)
            for field, val in node.get("inputs", {}).items()
        }
        normalized[new_id] = {
            "inputs": inputs,
            "class_type": class_type,
        }

    return normalized


def build_links_index(links: List[Any]) -> Dict[int, Dict[str, Any]]:
    index: Dict[int, Dict[str, Any]] = {}
    for link in links:
        if isinstance(link, dict):
            index[int(link["id"])] = link
        elif isinstance(link, list) and len(link) >= 6:
            index[int(link[0])] = {
                "id": link[0],
                "origin_id": link[1],
                "origin_slot": link[2],
                "target_id": link[3],
                "target_slot": link[4],
                "type": link[5],
            }
    return index


def resolve_subgraph_output(subgraph: Dict[str, Any], slot: int) -> Tuple[str, int]:
    outputs = subgraph.get("outputs", [])
    if slot >= len(outputs):
        raise ValueError(f"Subgraph output slot {slot} не найден")
    link_ids = outputs[slot].get("linkIds", [])
    if not link_ids:
        raise ValueError(f"Subgraph output slot {slot} не имеет linkIds")
    link_id = link_ids[0]
    for link in subgraph.get("links", []):
        if link.get("id") == link_id:
            return str(link["origin_id"]), int(link["origin_slot"])
    raise ValueError(f"Subgraph link {link_id} не найден")


def resolve_link_origin(
    link: Dict[str, Any],
    subgraph_wrappers: Dict[int, str],
    subgraphs_by_uuid: Dict[str, Dict[str, Any]],
) -> Tuple[str, int]:
    origin_id = link["origin_id"]
    origin_slot = int(link["origin_slot"])

    if origin_id in subgraph_wrappers:
        subgraph = subgraphs_by_uuid[subgraph_wrappers[origin_id]]
        return resolve_subgraph_output(subgraph, origin_slot)
    return str(origin_id), origin_slot


def collect_ui_graph(
    data: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Any], Dict[int, str], Dict[str, Dict[str, Any]]]:
    subgraphs = data.get("definitions", {}).get("subgraphs", [])
    subgraphs_by_uuid = {sg["id"]: sg for sg in subgraphs}
    subgraph_wrappers: Dict[int, str] = {}

    for node in data.get("nodes", []):
        node_type = node.get("type")
        if node_type in subgraphs_by_uuid:
            subgraph_wrappers[int(node["id"])] = node_type

    if subgraphs:
        primary = subgraphs[0]
        nodes = list(primary.get("nodes", []))
        links = list(primary.get("links", []))
        for node in data.get("nodes", []):
            node_id = int(node["id"])
            if node_id in subgraph_wrappers:
                continue
            if node.get("type") in SKIP_CLASS_TYPES:
                continue
            nodes.append(node)
        links.extend(data.get("links", []))
    else:
        nodes = list(data.get("nodes", []))
        links = list(data.get("links", []))

    return nodes, links, subgraph_wrappers, subgraphs_by_uuid


def convert_ui_node_to_api(
    node: Dict[str, Any],
    links_by_id: Dict[int, Dict[str, Any]],
    subgraph_wrappers: Dict[int, str],
    subgraphs_by_uuid: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    class_type = node.get("type", "")
    if class_type in SKIP_CLASS_TYPES:
        return None

    widgets_values = list(node.get("widgets_values") or [])
    widget_idx = 0
    inputs: Dict[str, Any] = {}

    for inp in node.get("inputs", []):
        name = inp["name"]
        link_id = inp.get("link")

        if link_id is not None:
            link = links_by_id[int(link_id)]
            origin_id = link["origin_id"]

            if origin_id == -10:
                if widget_idx < len(widgets_values):
                    inputs[name] = widgets_values[widget_idx]
                    widget_idx += 1
            else:
                ref_id, ref_slot = resolve_link_origin(link, subgraph_wrappers, subgraphs_by_uuid)
                inputs[name] = [ref_id, ref_slot]
        elif "widget" in inp:
            if widget_idx < len(widgets_values):
                inputs[name] = widgets_values[widget_idx]
                widget_idx += 1

    for field in EXTRA_WIDGET_FIELDS.get(class_type, []):
        if field not in inputs and widget_idx < len(widgets_values):
            inputs[field] = widgets_values[widget_idx]
            widget_idx += 1

    return {"class_type": class_type, "inputs": inputs}


def ui_to_api_workflow(data: Dict[str, Any]) -> Dict[str, Any]:
    ui_nodes, links, subgraph_wrappers, subgraphs_by_uuid = collect_ui_graph(data)
    links_by_id = build_links_index(links)
    workflow: Dict[str, Any] = {}

    for node in ui_nodes:
        api_node = convert_ui_node_to_api(node, links_by_id, subgraph_wrappers, subgraphs_by_uuid)
        if api_node is None:
            continue
        workflow[str(node["id"])] = api_node

    return workflow


def extract_models_from_ui_nodes(ui_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    for node in ui_nodes:
        class_type = node.get("type", "")
        for model in node.get("properties", {}).get("models", []):
            name = model.get("name", "")
            url = model.get("url", "")
            directory = model.get("directory", "")
            if not name:
                continue

            category = MODEL_DIR_TO_CATEGORY.get(directory, directory or "other")
            models.setdefault(category, {})
            models[category][name] = {
                "url": url,
                "path": f"models/{directory}/{name}" if directory else f"models/{name}",
                "description": f"{class_type} (узел {node.get('id')})",
            }

    return models


def merge_models(auto_models: Dict[str, Any], imported_models: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(auto_models)
    for category, entries in imported_models.items():
        merged.setdefault(category, {})
        for name, info in entries.items():
            existing = merged[category].get(name, {})
            merged[category][name] = {
                **existing,
                **info,
                "url": info.get("url") or existing.get("url", ""),
                "path": info.get("path") or existing.get("path", ""),
                "description": info.get("description") or existing.get("description", name),
            }
    return merged


def detect_source_format(data: Dict[str, Any], workflow: Dict[str, Any]) -> str:
    if is_ui_workflow(data):
        return "ui"
    if has_subgraph_ids(workflow):
        return "api_subgraph"
    return "api"


def prepare_workflow_data(data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    workflow, meta = split_workflow(data)

    if is_ui_workflow(data):
        ui_nodes, _, _, _ = collect_ui_graph(data)
        imported_models = extract_models_from_ui_nodes(ui_nodes)
        workflow = ui_to_api_workflow(data)
        if imported_models:
            meta["_models_imported"] = merge_models(meta.get("_models", {}), imported_models)
        source_format = "ui"
    elif is_api_workflow(data):
        source_format = "api_subgraph" if has_subgraph_ids(workflow) else "api"
        if has_subgraph_ids(workflow):
            workflow = normalize_api_workflow(workflow)
    else:
        raise ValueError(
            "Неизвестный формат workflow. Положите UI-экспорт ComfyUI (Save workflow) "
            "или API Format в папку workflow/."
        )

    return workflow, meta, source_format


def model_storage_path(class_type: str, field: str, filename: str, inputs: Dict[str, Any]) -> str:
    loader_info = LOADER_NODES.get(class_type)
    if not loader_info:
        return f"models/{filename}"

    _, _, default_dir = loader_info
    if class_type == "CLIPLoader":
        clip_type = inputs.get("type", "")
        default_dir = CLIP_TYPE_DIRS.get(str(clip_type), default_dir)

    return f"models/{default_dir}/{filename}"


def is_loader_node(node: Dict[str, Any]) -> bool:
    return node.get("class_type") in LOADER_NODES


def find_output_nodes(workflow: Dict[str, Any]) -> List[str]:
    return [
        node_id
        for node_id in sort_node_ids(list(workflow.keys()))
        if workflow[node_id].get("class_type") in OUTPUT_CLASS_TYPES
    ]


def auto_build_inputs(workflow: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
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


def auto_build_models(workflow: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    models = copy.deepcopy(existing) if existing else {}

    for node_id in sort_node_ids(list(workflow.keys())):
        node = workflow[node_id]
        class_type = node.get("class_type")
        loader_info = LOADER_NODES.get(class_type)
        if not loader_info:
            continue

        category, field, _ = loader_info
        filename = node.get("inputs", {}).get(field)
        if not isinstance(filename, str) or not filename:
            continue

        models.setdefault(category, {})
        if filename in models[category]:
            continue

        models[category][filename] = {
            "url": "",
            "path": model_storage_path(
                class_type,
                field,
                filename,
                node.get("inputs", {}),
            ),
            "description": f"{class_type} (узел {node_id})",
        }

    return models


def build_meta(workflow: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    output_nodes = find_output_nodes(workflow)
    imported_models = existing.get("_models_imported", existing.get("_models", {}))
    return {
        "_inputs": auto_build_inputs(workflow),
        "_output": {"node": output_nodes[0]} if output_nodes else existing.get("_output", {}),
        "_models": merge_models(auto_build_models(workflow, {}), imported_models),
    }


def load_workflow(source: Optional[Path] = None) -> LoadedWorkflow:
    path = resolve_workflow_source(source)

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Workflow не найден: {path}\n"
            f"Положите UI-экспорт ComfyUI в workflow/workflow.json"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ошибка парсинга {path}: {exc}") from exc

    workflow, meta, source_format = prepare_workflow_data(raw)
    built = build_meta(workflow, meta)

    if not built["_inputs"]:
        raise ValueError(f"Не удалось собрать параметры API из {path}")
    if not built["_output"].get("node"):
        raise ValueError(f"Не найден выходной узел (SaveImage и т.п.) в {path}")

    return LoadedWorkflow(
        source_path=path,
        workflow=workflow,
        inputs_config=built["_inputs"],
        output_config=built["_output"],
        models_info=built["_models"],
        source_format=source_format,
    )
