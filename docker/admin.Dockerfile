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

COPY admin /app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
