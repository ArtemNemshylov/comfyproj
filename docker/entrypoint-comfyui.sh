#!/usr/bin/env bash
# Стартовий скрипт comfyui-контейнера: довантажує моделі (якщо їх ще
# немає у примонтованому томі), генерує extra_model_paths.yaml так, щоб
# ComfyUI бачив LoRA з admin-контейнера, і запускає сервер.
set -euo pipefail

MODELS_DIR="/app/comfyui/models"
MOTION_DIR="/app/comfyui/custom_nodes/ComfyUI-AnimateDiff-Evolved/models"
ADMIN_DATA_MOUNT="${COMPF_ADMIN_DATA_MOUNT:-/admin-data}"

mkdir -p "$MODELS_DIR/checkpoints" "$MODELS_DIR/upscale_models" "$MOTION_DIR"

CKPT="$MODELS_DIR/checkpoints/v1-5-pruned-emaonly.safetensors"
if [ ! -f "$CKPT" ]; then
    echo "[entrypoint] Завантаження SD1.5 checkpoint (~4 ГБ)..."
    curl -L --fail -C - -o "$CKPT" \
        "https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly.safetensors"
else
    echo "[entrypoint] Checkpoint вже є, пропускаємо"
fi

MM_FILE="$MOTION_DIR/mm_sd_v15_v2.ckpt"
if [ ! -f "$MM_FILE" ]; then
    echo "[entrypoint] Завантаження AnimateDiff motion-модуля (~1.7 ГБ)..."
    curl -L --fail -C - -o "$MM_FILE" \
        "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt"
else
    echo "[entrypoint] Motion-модуль вже є, пропускаємо"
fi

UPSCALE_FILE="$MODELS_DIR/upscale_models/RealESRGAN_x4plus.pth"
if [ ! -f "$UPSCALE_FILE" ]; then
    echo "[entrypoint] Завантаження ESRGAN upscale-моделі (~64 МБ)..."
    curl -L --fail -C - -o "$UPSCALE_FILE" \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
else
    echo "[entrypoint] Upscale-модель вже є, пропускаємо"
fi

echo "[entrypoint] Пишемо extra_model_paths.yaml (LoRA з ${ADMIN_DATA_MOUNT}/loras)"
cat > /app/comfyui/extra_model_paths.yaml << YAML
compf_admin:
    base_path: ${ADMIN_DATA_MOUNT}
    loras: loras
YAML

echo "[entrypoint] Старт ComfyUI на 0.0.0.0:8188"
cd /app/comfyui
exec python main.py --listen 0.0.0.0 --port 8188
