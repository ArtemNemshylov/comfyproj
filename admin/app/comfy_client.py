"""
Тонкий клієнт до ComfyUI HTTP API.

ComfyUI не має простого REST-виклику "згенеруй фото" — треба відправити
повний workflow-граф (той самий, що складається з нод у його UI) як
JSON у POST /prompt, а потім опитувати GET /history/{id}, поки потрібна
нода (SaveImage / VHS_VideoCombine) не запише файл на диск. Саме це
і робить цей модуль.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Optional

import requests

from .config import COMFYUI_URL


class ComfyUIError(RuntimeError):
    pass


class ComfyUIClient:
    def __init__(self, base_url: str = COMFYUI_URL, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    def is_alive(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/system_stats", timeout=3)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def object_info(self) -> dict:
        """Список усіх нод, які реально зареєстровані в цьому ComfyUI
        (включно з кастомними). Корисно, якщо AnimateDiff-workflow
        падає з незнайомим node type — тут видно точну назву класу,
        яку встановив кастомний вузол."""
        r = requests.get(f"{self.base_url}/object_info", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        payload = {"prompt": workflow, "client_id": self.client_id}
        r = requests.post(f"{self.base_url}/prompt", json=payload, timeout=self.timeout)
        if r.status_code != 200:
            raise ComfyUIError(f"ComfyUI /prompt повернув {r.status_code}: {r.text}")
        data = r.json()
        if data.get("node_errors"):
            raise ComfyUIError(f"ComfyUI відхилив workflow (node_errors): {data['node_errors']}")
        if "error" in data:
            raise ComfyUIError(f"ComfyUI відхилив workflow: {data['error']}")
        return data["prompt_id"]

    def get_history(self, prompt_id: str) -> Optional[dict]:
        r = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=self.timeout)
        r.raise_for_status()
        return r.json().get(prompt_id)

    def wait_for_result(self, prompt_id: str, poll_interval: float = 2.0, max_wait: float = 1800.0) -> dict:
        """Блокуючий опит статусу задачі. Виклики йдуть з фонових Job'ів
        (див. training.py / routers/generate.py), тож блокування тут не
        морозить решту застосунку."""
        started = time.time()
        while True:
            history = self.get_history(prompt_id)
            if history is not None and history.get("outputs"):
                return history
            if history is not None:
                status = history.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyUIError(f"Виконання ComfyUI завершилось помилкою: {status}")
            if time.time() - started > max_wait:
                raise ComfyUIError("Таймаут очікування результату від ComfyUI")
            time.sleep(poll_interval)

    def fetch_output_file(self, filename: str, subfolder: str = "", file_type: str = "output") -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": file_type}
        r = requests.get(f"{self.base_url}/view", params=params, timeout=120)
        r.raise_for_status()
        return r.content

    @staticmethod
    def extract_saved_files(history: dict) -> list[dict]:
        """Дістає список файлів (images/gifs/videos), які будь-яка нода
        записала в outputs цього history-запису."""
        files: list[dict] = []
        outputs = history.get("outputs", {})
        for node_output in outputs.values():
            for key in ("images", "gifs", "videos"):
                for item in node_output.get(key, []) or []:
                    files.append(item)
        return files
