"""
Точка входу FastAPI-застосунку compfui-admin.

Запуск: uvicorn app.main:app --reload --port 8000  (з теки admin/,
зазвичай через scripts/run_all.sh у звичайному терміналі macOS).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config, progress
from .db import init_db
from .routers import characters, generate, video, instagram, freeform

app = FastAPI(title="compfui-admin", version="0.1.0-poc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _on_startup() -> None:
    init_db()
    progress.start_background_listener()


app.include_router(characters.router)
app.include_router(generate.router)
app.include_router(video.router)
app.include_router(instagram.router)
app.include_router(freeform.router)


@app.get("/api/health")
def health():
    from .comfy_client import ComfyUIClient
    comfy_alive = ComfyUIClient().is_alive()
    return {"ok": True, "comfyui_alive": comfy_alive, "comfyui_url": config.COMFYUI_URL}


@app.get("/api/generate/progress")
def generation_progress():
    return progress.generation_state


@app.get("/api/models/download-status")
def models_download_status():
    return progress.model_download_status()


# --- Статичні файли ---
# Згенеровані фото/відео та референсні фото — щоб фронтенд міг показати їх напряму по URL.
app.mount("/media/outputs", StaticFiles(directory=str(config.DATA_DIR / "outputs")), name="outputs")
app.mount("/media/characters", StaticFiles(directory=str(config.CHARACTERS_DIR)), name="characters-media")

# Сама адмінка (index.html/app.js/style.css) — монтуємо останньою, бо html=True перехоплює "/".
app.mount("/", StaticFiles(directory=str(config.BASE_DIR / "static"), html=True), name="static")
