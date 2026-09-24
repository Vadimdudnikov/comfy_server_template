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
    "LoadImage": ["image"],
    "RandomNoise": ["noise_seed"],
    "KSamplerSelect": ["sampler_name"],
    "CFGGuider": ["cfg"],
    "Flux2Scheduler": ["steps"],
    "EmptyFlux2LatentImage": ["batch_size"],
    "ImageScaleToTotalPixels": ["upscale_method", "megapixels", "resolution_steps"],
    "ModelSamplingAuraFlow": ["shift"],
    "StringConcatenate": ["string_b", "delimiter"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    "LoraLoaderModelOnly": ["lora_name", "strength_model"],
}

# Полный порядок widgets_values у узла (как в ComfyUI), чтобы корректно
# доставать EXTRA-поля, когда часть виджетов уже пришла через линк/-10.
WIDGET_ORDER: Dict[str, List[str]] = {
    "UNETLoader": ["unet_name", "weight_dtype"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "VAELoader": ["vae_name"],
    "CLIPTextEncode": ["text"],
    "SaveImage": ["filename_prefix"],
    "LoadImage": ["image", "upload"],
    "RandomNoise": ["noise_seed", "control_after_generate"],
    "KSamplerSelect": ["sampler_name"],
    "CFGGuider": ["cfg"],
    "Flux2Scheduler": ["steps", "width", "height"],
    "EmptyFlux2LatentImage": ["width", "height", "batch_size"],
    "ImageScaleToTotalPixels": ["upscale_method", "megapixels", "resolution_steps"],
    "ModelSamplingAuraFlow": ["shift"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    "LoraLoaderModelOnly": ["lora_name", "strength_model"],
}

SKIP_WIDGET_FIELDS = frozenset({
    "upload",
    "control_after_generate",
})


# Примитивные типы subgraph-входов берутся из widgets_values обёртки.
# IMAGE/MODEL/... — из внешних линков на обёртку.
SUBGRAPH_WIDGET_TYPES = frozenset({
    "INT",
    "FLOAT",
    "STRING",
    "BOOLEAN",
    "COMBO",
})

BYPASS_MODE = 4

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
    raw_data: Optional[Dict[str, Any]] = None


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


def _wrapper_linked_inputs(wrapper: Dict[str, Any]) -> Dict[str, int]:
    linked: Dict[str, int] = {}
    for inp in wrapper.get("inputs", []):
        link_id = inp.get("link")
        if link_id is not None:
            linked[inp["name"]] = int(link_id)
    return linked


def build_subgraph_bindings(
    wrapper: Dict[str, Any],
    subgraph: Dict[str, Any],
    parent_links: Dict[int, Dict[str, Any]],
    parent_bindings: Optional[Dict[int, Tuple[str, Any]]],
    subgraph_wrappers: Dict[int, str],
    subgraphs_by_uuid: Dict[str, Dict[str, Any]],
) -> Dict[int, Tuple[str, Any]]:
    """
    origin_slot (-10:N) → ('link', [node_id, slot]) | ('value', scalar)
    """
    bindings: Dict[int, Tuple[str, Any]] = {}
    linked = _wrapper_linked_inputs(wrapper)
    widgets = list(wrapper.get("widgets_values") or [])
    widget_idx = 0

    for slot, sg_inp in enumerate(subgraph.get("inputs", [])):
        name = sg_inp.get("name", "")
        sg_type = str(sg_inp.get("type", ""))

        if name in linked:
            link = parent_links[linked[name]]
            bindings[slot] = resolve_binding_origin(
                link,
                parent_bindings,
                subgraph_wrappers,
                subgraphs_by_uuid,
            )
            continue

        if sg_type in SUBGRAPH_WIDGET_TYPES or name not in linked:
            if widget_idx < len(widgets):
                bindings[slot] = ("value", widgets[widget_idx])
                widget_idx += 1
            else:
                bindings[slot] = ("value", None)

    return bindings


def resolve_binding_origin(
    link: Dict[str, Any],
    parent_bindings: Optional[Dict[int, Tuple[str, Any]]],
    subgraph_wrappers: Dict[int, str],
    subgraphs_by_uuid: Dict[str, Dict[str, Any]],
) -> Tuple[str, Any]:
    origin_id = link["origin_id"]
    origin_slot = int(link["origin_slot"])

    if origin_id == -10:
        if not parent_bindings or origin_slot not in parent_bindings:
            raise ValueError(f"Не найден binding для subgraph input slot {origin_slot}")
        return parent_bindings[origin_slot]

    if origin_id in subgraph_wrappers:
        subgraph = subgraphs_by_uuid[subgraph_wrappers[origin_id]]
        ref_id, ref_slot = resolve_subgraph_output(subgraph, origin_slot)
        return ("link", [ref_id, ref_slot])

    return ("link", [str(origin_id), origin_slot])


def resolve_link_to_input(
    link: Dict[str, Any],
    bindings: Optional[Dict[int, Tuple[str, Any]]],
    subgraph_wrappers: Dict[int, str],
    subgraphs_by_uuid: Dict[str, Dict[str, Any]],
) -> Any:
    kind, value = resolve_binding_origin(link, bindings, subgraph_wrappers, subgraphs_by_uuid)
    if kind == "link":
        return value
    return value


def convert_ui_node_to_api(
    node: Dict[str, Any],
    links_by_id: Dict[int, Dict[str, Any]],
    bindings: Optional[Dict[int, Tuple[str, Any]]],
    subgraph_wrappers: Dict[int, str],
    subgraphs_by_uuid: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    class_type = node.get("type", "")
    if class_type in SKIP_CLASS_TYPES:
        return None
    if class_type in subgraphs_by_uuid:
        return None

    widgets_values = list(node.get("widgets_values") or [])
    inputs: Dict[str, Any] = {}

    for inp in node.get("inputs", []):
        name = inp["name"]
        link_id = inp.get("link")

        if link_id is not None:
            link = links_by_id[int(link_id)]
            inputs[name] = resolve_link_to_input(
                link,
                bindings,
                subgraph_wrappers,
                subgraphs_by_uuid,
            )
        elif "widget" in inp:
            # Значение возьмём ниже через WIDGET_ORDER / fallback
            pass

    order = WIDGET_ORDER.get(class_type)
    if order and len(widgets_values) >= len(order):
        for field, value in zip(order, widgets_values):
            if field in SKIP_WIDGET_FIELDS or field in inputs:
                continue
            inputs[field] = value
    else:
        widget_idx = 0
        for inp in node.get("inputs", []):
            name = inp["name"]
            if name in inputs:
                if "widget" in inp:
                    widget_idx += 1
                continue
            if "widget" in inp and widget_idx < len(widgets_values):
                inputs[name] = widgets_values[widget_idx]
                widget_idx += 1
        for field in EXTRA_WIDGET_FIELDS.get(class_type, []):
            if field not in inputs and field not in SKIP_WIDGET_FIELDS and widget_idx < len(widgets_values):
                inputs[field] = widgets_values[widget_idx]
                widget_idx += 1

    return {"class_type": class_type, "inputs": inputs}


def _local_subgraph_wrappers(
    nodes: List[Dict[str, Any]],
    subgraphs_by_uuid: Dict[str, Dict[str, Any]],
) -> Dict[int, str]:
    wrappers: Dict[int, str] = {}
    for node in nodes:
        node_type = node.get("type")
        if node_type in subgraphs_by_uuid:
            wrappers[int(node["id"])] = node_type
    return wrappers


def expand_subgraph_into_workflow(
    workflow: Dict[str, Any],
    sg_uuid: str,
    wrapper: Dict[str, Any],
    parent_links: Dict[int, Dict[str, Any]],
    parent_bindings: Optional[Dict[int, Tuple[str, Any]]],
    subgraphs_by_uuid: Dict[str, Dict[str, Any]],
    parent_wrappers: Optional[Dict[int, str]] = None,
) -> None:
    subgraph = subgraphs_by_uuid[sg_uuid]
    nodes = list(subgraph.get("nodes", []))
    links_by_id = build_links_index(subgraph.get("links", []))
    local_wrappers = _local_subgraph_wrappers(nodes, subgraphs_by_uuid)
    # Линки на обёртку живут в родительском графе — резолвим через parent_wrappers.
    wrappers_for_bindings = parent_wrappers if parent_wrappers is not None else local_wrappers
    bindings = build_subgraph_bindings(
        wrapper,
        subgraph,
        parent_links,
        parent_bindings,
        wrappers_for_bindings,
        subgraphs_by_uuid,
    )

    for node in nodes:
        if int(node.get("mode", 0) or 0) == BYPASS_MODE:
            continue

        node_type = node.get("type", "")
        if node_type in SKIP_CLASS_TYPES:
            continue

        if node_type in subgraphs_by_uuid:
            expand_subgraph_into_workflow(
                workflow,
                node_type,
                node,
                links_by_id,
                bindings,
                subgraphs_by_uuid,
                parent_wrappers=local_wrappers,
            )
            continue

        api_node = convert_ui_node_to_api(
            node,
            links_by_id,
            bindings,
            local_wrappers,
            subgraphs_by_uuid,
        )
        if api_node is not None:
            workflow[str(node["id"])] = api_node


def collect_ui_graph(
    data: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Any], Dict[int, str], Dict[str, Dict[str, Any]]]:
    """Совместимость: возвращает активные top-level узлы для extract_models."""
    subgraphs = data.get("definitions", {}).get("subgraphs", [])
    subgraphs_by_uuid = {sg["id"]: sg for sg in subgraphs}
    subgraph_wrappers: Dict[int, str] = {}
    nodes: List[Dict[str, Any]] = []
    links = list(data.get("links", []))

    for node in data.get("nodes", []):
        if int(node.get("mode", 0) or 0) == BYPASS_MODE:
            continue
        node_type = node.get("type")
        if node_type in subgraphs_by_uuid:
            subgraph_wrappers[int(node["id"])] = node_type
            sg = subgraphs_by_uuid[node_type]
            nodes.extend(sg.get("nodes", []))
            links.extend(sg.get("links", []))
            continue
        if node_type in SKIP_CLASS_TYPES:
            continue
        nodes.append(node)

    return nodes, links, subgraph_wrappers, subgraphs_by_uuid


def ui_to_api_workflow(data: Dict[str, Any]) -> Dict[str, Any]:
    subgraphs = data.get("definitions", {}).get("subgraphs", [])
    subgraphs_by_uuid = {sg["id"]: sg for sg in subgraphs}
    top_links = build_links_index(data.get("links", []))
    top_wrappers = _local_subgraph_wrappers(data.get("nodes", []), subgraphs_by_uuid)
    workflow: Dict[str, Any] = {}

    for node in data.get("nodes", []):
        if int(node.get("mode", 0) or 0) == BYPASS_MODE:
            continue

        node_type = node.get("type", "")
        if node_type in SKIP_CLASS_TYPES:
            continue

        if node_type in subgraphs_by_uuid:
            expand_subgraph_into_workflow(
                workflow,
                node_type,
                node,
                top_links,
                None,
                subgraphs_by_uuid,
                parent_wrappers=top_wrappers,
            )
            continue

        api_node = convert_ui_node_to_api(
            node,
            top_links,
            None,
            top_wrappers,
            subgraphs_by_uuid,
        )
        if api_node is not None:
            workflow[str(node["id"])] = api_node

    return workflow


def _count_subgraph_image_inputs(subgraph: Dict[str, Any]) -> int:
    return sum(1 for inp in subgraph.get("inputs", []) if inp.get("type") == "IMAGE")


def _iter_top_links(data: Dict[str, Any]) -> List[Tuple[int, int]]:
    """(origin_id, target_id) для top-level links."""
    pairs: List[Tuple[int, int]] = []
    for link in data.get("links", []):
        if isinstance(link, list) and len(link) >= 6:
            pairs.append((int(link[1]), int(link[3])))
        elif isinstance(link, dict):
            pairs.append((int(link["origin_id"]), int(link["target_id"])))
    return pairs


def select_image_edit_path(data: Dict[str, Any], num_images: int) -> Dict[str, Any]:
    """
    Для UI-workflow с single/dual image-edit subgraph:
    включает нужный путь и bypass лишних LoadImage/SaveImage.
    """
    if not is_ui_workflow(data):
        return data

    subgraphs = {sg["id"]: sg for sg in data.get("definitions", {}).get("subgraphs", [])}
    if not subgraphs:
        return data

    data = copy.deepcopy(data)
    wrappers: List[Tuple[Dict[str, Any], int]] = []
    for node in data.get("nodes", []):
        node_type = node.get("type")
        if node_type not in subgraphs:
            continue
        image_count = _count_subgraph_image_inputs(subgraphs[node_type])
        if image_count >= 1:
            wrappers.append((node, image_count))

    if len(wrappers) < 2:
        return data

    target = 2 if num_images >= 2 else 1
    chosen = next((node for node, n_img in wrappers if n_img == target), None)
    if chosen is None:
        chosen = min(wrappers, key=lambda item: abs(item[1] - target))[0]

    chosen_id = int(chosen["id"])
    wrapper_ids = {int(node["id"]) for node, _ in wrappers}

    feeds_save: Dict[int, List[int]] = {}
    loads_into: Dict[int, List[int]] = {}
    load_ids = {
        int(node["id"])
        for node in data.get("nodes", [])
        if node.get("type") == "LoadImage"
    }
    save_ids = {
        int(node["id"])
        for node in data.get("nodes", [])
        if node.get("type") in OUTPUT_CLASS_TYPES
    }

    for origin_id, target_id in _iter_top_links(data):
        if origin_id in wrapper_ids and target_id in save_ids:
            feeds_save.setdefault(origin_id, []).append(target_id)
        if origin_id in load_ids and target_id in wrapper_ids:
            loads_into.setdefault(target_id, []).append(origin_id)

    active_saves = set(feeds_save.get(chosen_id, []))
    active_loads = set(loads_into.get(chosen_id, []))

    for node in data.get("nodes", []):
        nid = int(node["id"])
        node_type = node.get("type")
        if node_type in subgraphs and _count_subgraph_image_inputs(subgraphs[node_type]) >= 1:
            node["mode"] = 0 if nid == chosen_id else BYPASS_MODE
        elif node_type == "LoadImage" and nid in load_ids:
            # Неактивные LoadImage не должны попадать в prompt — ComfyUI валидирует файлы
            node["mode"] = 0 if nid in active_loads else BYPASS_MODE
        elif node_type in OUTPUT_CLASS_TYPES and nid in save_ids:
            # SaveImage, связанные с image-edit путями
            related = any(nid in feeds_save.get(wid, []) for wid in wrapper_ids)
            if related:
                node["mode"] = 0 if nid in active_saves else BYPASS_MODE

    return data


def max_reference_images(data: Dict[str, Any]) -> int:
    if not is_ui_workflow(data):
        return 0
    subgraphs = data.get("definitions", {}).get("subgraphs", [])
    counts = [_count_subgraph_image_inputs(sg) for sg in subgraphs]
    return max(counts) if counts else 0


def build_workflow_variant(
    raw: Dict[str, Any],
    num_images: int,
) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    """Готовит API-workflow под N референсов (single/dual path)."""
    selected = select_image_edit_path(raw, num_images) if is_ui_workflow(raw) else raw
    return prepare_workflow_data(selected)


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
        # Также модели из вложенных узлов активных subgraph
        for node in data.get("nodes", []):
            if int(node.get("mode", 0) or 0) == BYPASS_MODE:
                continue
            props_models = node.get("properties", {}).get("models")
            if props_models:
                imported_models = merge_models(
                    imported_models,
                    extract_models_from_ui_nodes([node]),
                )
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
            # Линк, ошибочно попавший как строка вида "nearest-exact" на IMAGE-вход — не API-параметр,
            # если у узла поле image и class не LoadImage (обрабатывается ниже отдельно).
            key = f"{node_id}_{field}"
            inputs[key] = {"node": node_id, "field": field}

    # Удобные алиасы для референсов: image, image_1, ...
    image_nodes = [
        node_id
        for node_id in sort_node_ids(list(workflow.keys()))
        if workflow[node_id].get("class_type") == "LoadImage"
        and "image" in workflow[node_id].get("inputs", {})
    ]
    for index, node_id in enumerate(image_nodes):
        alias = "image" if index == 0 else f"image_{index}"
        inputs[alias] = {"node": node_id, "field": "image"}
        inputs.setdefault(f"{node_id}_image", {"node": node_id, "field": "image"})

    if image_nodes:
        inputs["images"] = {
            "type": "image_array",
            "field": "image",
            "nodes": image_nodes,
            "max_items": len(image_nodes),
        }

    # Алиас prompt для текстового промпта
    text_nodes = [
        node_id
        for node_id in sort_node_ids(list(workflow.keys()))
        if workflow[node_id].get("class_type") == "CLIPTextEncode"
        and isinstance(workflow[node_id].get("inputs", {}).get("text"), str)
    ]
    if text_nodes and "prompt" not in inputs:
        inputs["prompt"] = {"node": text_nodes[0], "field": "text"}

    seed_nodes = [
        node_id
        for node_id in sort_node_ids(list(workflow.keys()))
        if workflow[node_id].get("class_type") == "RandomNoise"
        and "noise_seed" in workflow[node_id].get("inputs", {})
    ]
    if seed_nodes and "seed" not in inputs:
        inputs["seed"] = {"node": seed_nodes[0], "field": "noise_seed"}

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

    # По умолчанию single-image путь (1 референс)
    selected = select_image_edit_path(raw, 1) if is_ui_workflow(raw) else raw
    workflow, meta, source_format = prepare_workflow_data(selected)
    built = build_meta(workflow, meta)

    if not built["_inputs"]:
        raise ValueError(f"Не удалось собрать параметры API из {path}")
    if not built["_output"].get("node"):
        raise ValueError(f"Не найден выходной узел (SaveImage и т.п.) в {path}")

    # В метаданных images отразим максимальное число референсов из UI
    max_images = max_reference_images(raw) if is_ui_workflow(raw) else 0
    if max_images > 0:
        image_nodes = [
            node_id
            for node_id in sort_node_ids(list(workflow.keys()))
            if workflow[node_id].get("class_type") == "LoadImage"
        ]
        built["_inputs"]["images"] = {
            "type": "image_array",
            "field": "image",
            "nodes": image_nodes,
            "max_items": max_images,
            "min_items": 1,
        }

    return LoadedWorkflow(
        source_path=path,
        workflow=workflow,
        inputs_config=built["_inputs"],
        output_config=built["_output"],
        models_info=built["_models"],
        source_format=source_format,
        raw_data=raw if is_ui_workflow(raw) else None,
    )
