#!/usr/bin/env bash
# Встановлення справжнього ComfyUI + кастомних нод + моделей.
#
# ВАЖЛИВО: запускайте цей скрипт у ЗВИЧАЙНОМУ терміналі macOS
# (Terminal.app / iTerm), А НЕ через Cowork/Claude Desktop-пісочницю —
# лише звичайний термінал macOS має доступ до Metal (MPS-прискорення
# на Apple Silicon) і не обмежений allowlist'ом до huggingface.co.
#
# Використання:
#   cd /Users/Artem/PycharmProjects/compfproject
#   bash scripts/setup_comfyui.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY_DIR="$PROJECT_ROOT/comfyui"

echo "== 1/6: Клонування ComfyUI =="
if [ ! -d "$COMFY_DIR" ]; then
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
else
    echo "ComfyUI вже склоновано в $COMFY_DIR, пропускаємо"
fi

echo "== 2/6: Віртуальне середовище ComfyUI =="
cd "$COMFY_DIR"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt

echo "== 3/6: Кастомні ноди (AnimateDiff-Evolved + VideoHelperSuite) =="
mkdir -p custom_nodes
if [ ! -d "custom_nodes/ComfyUI-AnimateDiff-Evolved" ]; then
    git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git custom_nodes/ComfyUI-AnimateDiff-Evolved
fi
if [ ! -d "custom_nodes/ComfyUI-VideoHelperSuite" ]; then
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git custom_nodes/ComfyUI-VideoHelperSuite
fi
pip install -r custom_nodes/ComfyUI-AnimateDiff-Evolved/requirements.txt 2>/dev/null || true
pip install -r custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt 2>/dev/null || true

echo "== 4/6: extra_model_paths.yaml (щоб ComfyUI бачив LoRA з admin/data/loras) =="
cp -f "$PROJECT_ROOT/scripts/extra_model_paths.yaml" "$COMFY_DIR/extra_model_paths.yaml"

echo "== 5/7: Базовий чекпоінт SD1.5 (~4 ГБ) =="
mkdir -p models/checkpoints
CKPT="models/checkpoints/v1-5-pruned-emaonly.safetensors"
if [ ! -f "$CKPT" ]; then
    curl -L --fail -o "$CKPT" \
        "https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly.safetensors" \
        || echo "!! Завантаження не вдалось. Завантажте вручну будь-який SD1.5 .safetensors і покладіть у $CKPT"
else
    echo "Чекпоінт вже є, пропускаємо"
fi

echo "== 6/7: AnimateDiff motion-модуль (~1.7 ГБ) =="
MM_DIR="custom_nodes/ComfyUI-AnimateDiff-Evolved/models"
mkdir -p "$MM_DIR"
MM_FILE="$MM_DIR/mm_sd_v15_v2.ckpt"
if [ ! -f "$MM_FILE" ]; then
    curl -L --fail -o "$MM_FILE" \
        "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt" \
        || echo "!! Завантаження не вдалось. Завантажте вручну і покладіть у $MM_FILE"
else
    echo "Motion-модуль вже є, пропускаємо"
fi

echo "== 7/7: ESRGAN upscale-модель для максимальної якості фото (~64 МБ) =="
mkdir -p models/upscale_models
UPSCALE_FILE="models/upscale_models/RealESRGAN_x4plus.pth"
if [ ! -f "$UPSCALE_FILE" ]; then
    curl -L --fail -o "$UPSCALE_FILE" \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
        || echo "!! Завантаження не вдалось. Завантажте вручну і покладіть у $UPSCALE_FILE"
else
    echo "Upscale-модель вже є, пропускаємо"
fi

echo
echo "Готово. Перевірка файлів:"
ls -lh models/checkpoints 2>/dev/null || true
ls -lh "$MM_DIR" 2>/dev/null || true
ls -lh models/upscale_models 2>/dev/null || true
echo
echo "Далі:  bash scripts/setup_admin.sh"
echo "Потім: bash scripts/run_all.sh"
