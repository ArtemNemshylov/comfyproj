#!/usr/bin/env bash
# Запускає ComfyUI + адмінку у фоні. Запускати у звичайному терміналі macOS.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY_DIR="$PROJECT_ROOT/comfyui"
ADMIN_DIR="$PROJECT_ROOT/admin"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "Запуск ComfyUI на http://127.0.0.1:8188 (лог: logs/comfyui.log)..."
( cd "$COMFY_DIR" && source .venv/bin/activate && \
  python main.py --force-fp16 --listen 127.0.0.1 --port 8188 \
  > "$LOG_DIR/comfyui.log" 2>&1 & echo $! > "$LOG_DIR/comfyui.pid" )

echo "Запуск адмінки на http://127.0.0.1:8000 (лог: logs/admin.log)..."
( cd "$ADMIN_DIR" && source .venv/bin/activate && \
  uvicorn app.main:app --host 127.0.0.1 --port 8000 \
  > "$LOG_DIR/admin.log" 2>&1 & echo $! > "$LOG_DIR/admin.pid" )

sleep 2
echo
echo "Відкрийте адмінку:  http://127.0.0.1:8000"
echo "ComfyUI напряму:    http://127.0.0.1:8188"
echo "Щоб зупинити:       bash scripts/stop_all.sh"
