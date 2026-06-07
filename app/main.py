import asyncio
import os
import threading
import time
import urllib.request
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .tasks import get_client, reload_client, run_pipeline

app = FastAPI(title="Comfy Server Template")

COMFYUI_URL = os.getenv("COMFYUI_URL", "127.0.0.1:8188")
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(os.getenv("STATIC_DIR", BASE_DIR / "static"))
STATIC_URL = "/static"

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


class Job:
    def __init__(self, job_id: str, inputs: Dict[str, Any], priority: str, future: "asyncio.Future"):
        self.id = job_id
        self.inputs = inputs
        self.priority = priority
        self.future = future


HIGH_Q: deque[Job] = deque()
LOW_Q: deque[Job] = deque()
TASKS: Dict[str, Dict[str, Any]] = {}
MAIN_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _now_ts() -> float:
    return time.time()


def _build_static_url(file_path: str) -> str:
    return f"{STATIC_URL}/{Path(file_path).name}"


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
                TASKS[job.id]["file"] = out_path
                TASKS[job.id]["url"] = _build_static_url(out_path)
                TASKS[job.id]["status"] = "completed"
                TASKS[job.id]["updated_at"] = _now_ts()
                if MAIN_LOOP and not job.future.done():
                    MAIN_LOOP.call_soon_threadsafe(job.future.set_result, {
                        "file": out_path,
                        "url": TASKS[job.id]["url"],
                    })
            except Exception as e:
                TASKS[job.id]["status"] = "failed"
                TASKS[job.id]["error"] = str(e)
                TASKS[job.id]["updated_at"] = _now_ts()
                if MAIN_LOOP and not job.future.done():
                    MAIN_LOOP.call_soon_threadsafe(job.future.set_exception, e)
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
    """Перечитать workflow_template.json после замены файла."""
    reload_client()
    return {"status": "ok", "inputs": get_client().get_inputs_config()}


@app.post("/run")
async def run(req: RunRequest):
    job_id = uuid.uuid4().hex
    priority = (req.priority or "low").lower()

    TASKS[job_id] = {
        "status": "queued",
        "priority": priority,
        "file": None,
        "url": None,
        "error": None,
        "inputs": req.inputs,
        "created_at": _now_ts(),
        "updated_at": _now_ts(),
    }

    assert MAIN_LOOP is not None
    fut: asyncio.Future = MAIN_LOOP.create_future()
    job = Job(job_id, req.inputs, priority, fut)

    if priority == "high":
        HIGH_Q.append(job)
    else:
        LOW_Q.append(job)

    if req.wait:
        try:
            await fut
            return {
                "id": job_id,
                "status": TASKS[job_id]["status"],
                "file": TASKS[job_id]["file"],
                "url": TASKS[job_id]["url"],
            }
        except Exception as e:
            return {"id": job_id, "status": "failed", "error": str(e)}

    return {"id": job_id, "status": "queued"}


@app.get("/status/{task_id}")
def status(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        return {"status": "not_found"}
    return {
        "id": task_id,
        "status": task.get("status"),
        "file": task.get("file"),
        "url": task.get("url"),
        "error": task.get("error"),
        "inputs": task.get("inputs"),
    }
