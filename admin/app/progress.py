"""
Два незалежні джерела "живого" прогресу для фронтенду:

1. Завантаження файлів моделі (Z-Image Turbo та подібні) — просто порівнюємо
   поточний розмір файлу на диску (спільний том з comfyui-контейнером) з
   очікуваним розміром із config.ZIMAGE_EXPECTED_FILES. Працює незалежно від
   того, хто саме якає файл (curl у entrypoint контейнера чи хтось вручну).

2. Прогрес генерації (крок семплера) — ComfyUI розсилає `progress`-події всім
   підключеним websocket-клієнтам (PromptServer.send_sync, без фільтра по
   client_id), тож досить одного спільного фонового слухача на весь застосунок
   замість окремого з'єднання на кожен запит генерації.
"""
from __future__ import annotations

import asyncio
import json
import logging

import websockets

from . import config

log = logging.getLogger(__name__)

generation_state = {"prompt_id": None, "value": 0, "max": 0, "running": False}

_ws_url = f"ws://{config.COMFYUI_HOST}:{config.COMFYUI_PORT}/ws?clientId=admin-progress-listener"


async def _listen_forever() -> None:
    while True:
        try:
            async with websockets.connect(_ws_url, open_timeout=5) as ws:
                log.info("progress listener: connected to %s", _ws_url)
                async for raw in ws:
                    if not isinstance(raw, str):
                        continue
                    try:
                        msg = json.loads(raw)
                    except ValueError:
                        continue
                    mtype = msg.get("type")
                    data = msg.get("data", {})
                    if mtype == "progress":
                        generation_state["prompt_id"] = data.get("prompt_id")
                        generation_state["value"] = data.get("value", 0)
                        generation_state["max"] = data.get("max", 0)
                        generation_state["running"] = True
                    elif mtype == "executing" and data.get("node") is None:
                        # node=None -> виконання цього prompt_id завершилось
                        generation_state["running"] = False
                        generation_state["value"] = 0
                        generation_state["max"] = 0
        except Exception as exc:
            log.warning("progress listener: disconnected (%s), reconnecting in 3s", exc)
            await asyncio.sleep(3)


def start_background_listener() -> None:
    asyncio.get_event_loop().create_task(_listen_forever())


def model_download_status() -> list[dict]:
    result = []
    for label, path, expected_bytes in config.ZIMAGE_EXPECTED_FILES:
        actual = path.stat().st_size if path.exists() else 0
        done = actual >= expected_bytes
        result.append({
            "label": label,
            "filename": path.name,
            "bytes": actual,
            "expected_bytes": expected_bytes,
            "pct": min(100, round(100 * actual / expected_bytes)) if expected_bytes else 0,
            "done": done,
        })
    return result
