# ComfyUI + AnimateDiff-Evolved + VideoHelperSuite.
# GPU: pip-торч сам тягне потрібні CUDA-бібліотеки (nvidia-cuda-runtime,
# cudnn, cublas тощо як залежності), тому окремий nvidia/cuda базовий
# образ не потрібен — досить NVIDIA-драйвера і nvidia-container-toolkit
# на ХОСТІ (не тут) + `--gpus all` / deploy.resources у compose.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git comfyui

WORKDIR /app/comfyui
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision torchaudio \
    && pip install --no-cache-dir -r requirements.txt

RUN mkdir -p custom_nodes \
    && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git custom_nodes/ComfyUI-AnimateDiff-Evolved \
    && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git custom_nodes/ComfyUI-VideoHelperSuite \
    && (pip install --no-cache-dir -r custom_nodes/ComfyUI-AnimateDiff-Evolved/requirements.txt || true) \
    && (pip install --no-cache-dir -r custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt || true)

COPY docker/entrypoint-comfyui.sh /app/entrypoint-comfyui.sh
RUN chmod +x /app/entrypoint-comfyui.sh

EXPOSE 8188
ENTRYPOINT ["/app/entrypoint-comfyui.sh"]
