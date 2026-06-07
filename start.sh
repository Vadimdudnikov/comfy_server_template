#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOGS_DIR="${LOGS_DIR:-$SCRIPT_DIR/logs}"
mkdir -p "$LOGS_DIR"

COMFYUI_DIR="${COMFYUI_DIR:-/home/ComfyUI}"
COMFYUI_PORT="${COMFYUI_PORT:-8188}"
SERVICE_PORT="${SERVICE_PORT:-8000}"
COMFYUI_URL="${COMFYUI_URL:-127.0.0.1:$COMFYUI_PORT}"

export COMFYUI_URL
export COMFYUI_BASE_DIR="$COMFYUI_DIR"
export WORKFLOW_PATH="${WORKFLOW_PATH:-$SCRIPT_DIR/app/workflow.json}"
export STATIC_DIR="${STATIC_DIR:-$SCRIPT_DIR/static}"

echo "== ComfyUI =="
if [ ! -d "$COMFYUI_DIR" ]; then
  echo "Клонирую ComfyUI в $COMFYUI_DIR..."
  git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
  pip install -r "$COMFYUI_DIR/requirements.txt"
else
  echo "ComfyUI уже установлен: $COMFYUI_DIR"
fi

echo "Запуск ComfyUI на порту $COMFYUI_PORT..."
python "$COMFYUI_DIR/main.py" --listen 0.0.0.0 --port "$COMFYUI_PORT" > "$LOGS_DIR/comfyui.log" 2>&1 &

COMFYUI_CHECK_URL="http://127.0.0.1:$COMFYUI_PORT/system_stats"
COMFYUI_READY_MAX_WAIT="${COMFYUI_READY_MAX_WAIT:-600}"
COMFYUI_READY_INTERVAL="${COMFYUI_READY_INTERVAL:-3}"
elapsed=0
until curl -sf "$COMFYUI_CHECK_URL" >/dev/null 2>&1; do
  sleep "$COMFYUI_READY_INTERVAL"
  elapsed=$((elapsed + COMFYUI_READY_INTERVAL))
  if [ "$elapsed" -ge "$COMFYUI_READY_MAX_WAIT" ]; then
    echo "ComfyUI не ответил за ${COMFYUI_READY_MAX_WAIT}с. Лог: $LOGS_DIR/comfyui.log"
    exit 1
  fi
done
echo "ComfyUI готов (${elapsed}с)"

echo "== Зависимости сервиса =="
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "== Модели =="
python "$SCRIPT_DIR/download_models.py" || echo "Предупреждение: не все модели скачаны"

echo "== Comfy Service =="
uvicorn app.main:app --host 0.0.0.0 --port "$SERVICE_PORT" --workers 1
