# compfui-admin backend (FastAPI + тренування LoRA).
# GPU так само через pip-торч, без окремого nvidia/cuda базового образу.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY admin/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ai-toolkit (ostris) — тренування LoRA для Z-Image Turbo (наш diffusers+peft
# скрипт вище вміє лише SD1.5; Z-Image — інша архітектура, DiT+Qwen-енкодер).
# Навмисно ПЕРЕД `COPY admin /app` (яка змінюється щоразу при правці коду) —
# так зміни коду адмінки інвалідують лише останній COPY-шар, а не цей важкий
# git-clone+pip-install.
RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git /app/ai-toolkit \
    && pip install --no-cache-dir -r /app/ai-toolkit/requirements.txt \
    && pip install --no-cache-dir torchaudio

# ai-toolkit тягне opencv-python (cv2, треба libGL.so.1) і Triton-ядра
# (треба C-компілятор для JIT — та сама проблема, що й у comfyui-образі).
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY admin /app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
