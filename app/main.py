import asyncio
import json
import os
import threading
import time
import urllib.request
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .tasks import get_client, reload_client, run_pipeline

app = FastAPI(title="Comfy Server Template")

COMFYUI_URL = os.getenv("COMFYUI_URL", "127.0.0.1:8188")
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(os.getenv("STATIC_DIR", BASE_DIR / "static"))
STATIC_URL = "/static"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
WEBHOOK_TIMEOUT = float(os.getenv("WEBHOOK_TIMEOUT", "15"))

STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount(STATIC_URL, StaticFiles(directory=STATIC_DIR), name="static")


def check_comfyui_connection() -> bool:
    try:
        url = f"http://{COMFYUI_URL}/system_stats"
        urllib.request.urlopen(urllib.request.Request(url), timeout=5)
        return True
    except Exception as e:
        print(f"ComfyUI недоступен ({COMFYUI_URL}): {e}", flush=True)
        return False


class RunRequest(BaseModel):
    inputs: Dict[str, Any] = Field(default_factory=dict)
    priority: Optional[str] = "low"
    wait: Optional[bool] = True
    webhook_url: Optional[str] = None

    @field_validator("webhook_url")
    @classmethod
    def validate_webhook_url(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        url = value.strip()
        if not url:
            return None
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("webhook_url должен быть http(s)://...")
        return url


class Job:
    def __init__(
        self,
        job_id: str,
        inputs: Dict[str, Any],
        priority: str,
        future: "asyncio.Future",
        webhook_url: Optional[str] = None,
    ):
        self.id = job_id
        self.inputs = inputs
        self.priority = priority
        self.future = future
        self.webhook_url = webhook_url


HIGH_Q: deque[Job] = deque()
LOW_Q: deque[Job] = deque()
TASKS: Dict[str, Dict[str, Any]] = {}
MAIN_LOOP: Optional[asyncio.AbstractEventLoop] = None


STATUS_MAP = {
    "queued": "QUEUED",
    "processing": "PROCESSING",
    "completed": "COMPLETED",
    "failed": "FAILED",
    "not_found": "NOT_FOUND",
}


def _now_ts() -> float:
    return time.time()


def _build_static_url(file_path: str) -> str:
    return f"{STATIC_URL}/{Path(file_path).name}"


def _is_unusable_host(host: str) -> bool:
    host = host.lower().split(":")[0]
    return host in {"", "0.0.0.0", "127.0.0.1", "localhost", "::", "::1"}


def _resolve_public_base(request: Optional[Request] = None) -> str:
    """Публичный origin сервиса: из заголовков прокси / Host запроса."""
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    if request is None:
        return ""

    headers = request.headers

    # RFC 7239: Forwarded: proto=https;host=example.com
    forwarded = headers.get("forwarded") or ""
    fwd_host = ""
    fwd_proto = ""
    if forwarded:
        first = forwarded.split(",")[0]
        for part in first.split(";"):
            part = part.strip()
            if part.lower().startswith("host="):
                fwd_host = part.split("=", 1)[1].strip().strip('"')
            elif part.lower().startswith("proto="):
                fwd_proto = part.split("=", 1)[1].strip().strip('"')

    host = (
        fwd_host
        or (headers.get("x-forwarded-host") or "").split(",")[0].strip()
        or (headers.get("x-original-host") or "").split(",")[0].strip()
        or (headers.get("host") or "").strip()
    )
    proto = (
        fwd_proto
        or (headers.get("x-forwarded-proto") or "").split(",")[0].strip()
        or (headers.get("x-forwarded-scheme") or "").split(",")[0].strip()
        or (headers.get("x-scheme") or "").split(",")[0].strip()
    )

    if host and not _is_unusable_host(host):
        if not proto:
            # RunPod / публичные прокси почти всегда https
            if "runpod.net" in host.lower() or "runpod.ai" in host.lower():
                proto = "https"
            else:
                proto = request.url.scheme or "http"
        return f"{proto}://{host}".rstrip("/")

    base = str(request.base_url).rstrip("/")
    parsed = urlparse(base)
    if parsed.hostname and not _is_unusable_host(parsed.hostname):
        return base
    return ""


def _image_url(relative_url: Optional[str], public_base: Optional[str] = None) -> Optional[str]:
    if not relative_url:
        return None
    if relative_url.startswith("http://") or relative_url.startswith("https://"):
        return relative_url

    base = (public_base or "").rstrip("/")
    if not base:
        return None
    path = relative_url if relative_url.startswith("/") else f"/{relative_url}"
    return f"{base}{path}"


def _public_status(status: Optional[str]) -> str:
    if not status:
        return "UNKNOWN"
    return STATUS_MAP.get(str(status).lower(), str(status).upper())


def _ensure_task_public_base(job_id: str, request: Optional[Request] = None) -> None:
    task = TASKS.get(job_id)
    if not task:
        return
    resolved = _resolve_public_base(request)
    if not resolved:
        return
    current = (task.get("public_base_url") or "").rstrip("/")
    if not current or _is_unusable_host(urlparse(current).hostname or ""):
        task["public_base_url"] = resolved


def _task_payload(job_id: str, request: Optional[Request] = None) -> Dict[str, Any]:
    task = TASKS.get(job_id)
    if not task:
        return {"id": job_id, "status": "NOT_FOUND"}

    _ensure_task_public_base(job_id, request)

    status = _public_status(task.get("status"))
    payload: Dict[str, Any] = {
        "id": job_id,
        "status": status,
    }

    if status == "COMPLETED":
        image_url = task.get("image_url") or _image_url(
            task.get("url"),
            task.get("public_base_url"),
        )
        if image_url and not image_url.startswith("http"):
            image_url = _image_url(task.get("url"), task.get("public_base_url"))
        if not image_url:
            payload["status"] = "FAILED"
            payload["error"] = "Не удалось определить публичный URL сервиса для image_url"
        else:
            task["image_url"] = image_url
            payload["output"] = {"image_url": image_url}
    elif status == "FAILED":
        payload["error"] = task.get("error")

    return payload


def _send_webhook(webhook_url: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=WEBHOOK_TIMEOUT) as response:
            print(
                f"Webhook OK {webhook_url} → {response.status} (job {payload.get('id')})",
                flush=True,
            )
    except Exception as e:
        print(f"Webhook failed {webhook_url} (job {payload.get('id')}): {e}", flush=True)


def _notify_webhook(job: Job) -> None:
    if not job.webhook_url:
        return
    payload = _task_payload(job.id)
    threading.Thread(
        target=_send_webhook,
        args=(job.webhook_url, payload),
        daemon=True,
    ).start()


def _worker_loop():
    while True:
        job: Optional[Job] = None
        try:
            if HIGH_Q:
                job = HIGH_Q.popleft()
            elif LOW_Q:
                job = LOW_Q.popleft()
            else:
                time.sleep(0.005)
                continue

            TASKS[job.id]["status"] = "processing"
            TASKS[job.id]["updated_at"] = _now_ts()

            try:
                out_path = run_pipeline(job.inputs, filename=f"{uuid.uuid4().hex}.png")
                relative = _build_static_url(out_path)
                TASKS[job.id]["file"] = out_path
                TASKS[job.id]["url"] = relative
                TASKS[job.id]["image_url"] = _image_url(
                    relative,
                    TASKS[job.id].get("public_base_url"),
                )
                TASKS[job.id]["status"] = "completed"
                TASKS[job.id]["updated_at"] = _now_ts()
                if MAIN_LOOP and not job.future.done():
                    MAIN_LOOP.call_soon_threadsafe(job.future.set_result, {
                        "file": out_path,
                        "url": relative,
                        "image_url": TASKS[job.id]["image_url"],
                    })
            except Exception as e:
                TASKS[job.id]["status"] = "failed"
                TASKS[job.id]["error"] = str(e)
                TASKS[job.id]["updated_at"] = _now_ts()
                if MAIN_LOOP and not job.future.done():
                    MAIN_LOOP.call_soon_threadsafe(job.future.set_exception, e)
            finally:
                _notify_webhook(job)
        except Exception:
            time.sleep(0.01)


@app.on_event("startup")
async def _on_startup():
    global MAIN_LOOP
    MAIN_LOOP = asyncio.get_event_loop()

    from .model_downloader import ensure_models_ready

    ensure_models_ready()
    check_comfyui_connection()

    threading.Thread(target=_worker_loop, daemon=True).start()


@app.get("/health")
def health():
    if not check_comfyui_connection():
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "comfyui_unreachable", "comfyui_url": COMFYUI_URL},
        )
    return {"status": "healthy", "comfyui": "ok"}


@app.get("/workflow/inputs")
def workflow_inputs():
    """Список параметров из секции _inputs текущего workflow."""
    return {"inputs": get_client().get_inputs_config()}


@app.post("/workflow/reload")
def workflow_reload():
    """Перечитать workflow из папки workflow/ после замены файла."""
    reload_client()
    return {"status": "ok", "inputs": get_client().get_inputs_config()}


@app.post("/run")
async def run(req: RunRequest, request: Request):
    job_id = uuid.uuid4().hex
    priority = (req.priority or "low").lower()
    public_base = _resolve_public_base(request)
    if public_base:
        print(f"public_base_url={public_base}", flush=True)

    TASKS[job_id] = {
        "status": "queued",
        "priority": priority,
        "file": None,
        "url": None,
        "image_url": None,
        "error": None,
        "inputs": req.inputs,
        "webhook_url": req.webhook_url,
        "public_base_url": public_base,
        "created_at": _now_ts(),
        "updated_at": _now_ts(),
    }

    assert MAIN_LOOP is not None
    fut: asyncio.Future = MAIN_LOOP.create_future()
    job = Job(job_id, req.inputs, priority, fut, webhook_url=req.webhook_url)

    if priority == "high":
        HIGH_Q.append(job)
    else:
        LOW_Q.append(job)

    if req.wait:
        try:
            await fut
            return _task_payload(job_id, request)
        except Exception as e:
            return {
                "id": job_id,
                "status": "FAILED",
                "error": str(e),
            }

    return {"id": job_id, "status": "QUEUED"}


@app.get("/status/{task_id}")
def status(task_id: str, request: Request):
    return _task_payload(task_id, request)
