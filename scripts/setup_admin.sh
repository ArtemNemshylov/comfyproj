#!/usr/bin/env bash
# Віртуальне середовище для адмінки (FastAPI backend + скрипт тренування LoRA).
# Запускати у звичайному терміналі macOS (потрібен реальний pip install torch тощо).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT/admin"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Готово. Активація вручну: source admin/.venv/bin/activate"
