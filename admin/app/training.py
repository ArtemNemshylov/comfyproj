"""
Запуск LoRA-тренування як фонового підпроцесу + читання його прогресу.

Сам процес тренування живе в training/train_lora.py — окремому скрипті
поза FastAPI-процесом (щоб важке навантаження PyTorch не блокувало
event loop і щоб його можна було спостерігати/вбити незалежно).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sqlmodel import Session

from . import config
from .models import (
    Character, CharacterVersion, Job, JobType, JobStatus, VersionStatus, CharacterStatus,
)

TRAIN_SCRIPT = config.BASE_DIR / "training" / "train_lora_zimage.py"


def _progress_file(job_id: int) -> Path:
    return config.JOBS_LOG_DIR / f"job_{job_id}.json"


def _log_file(job_id: int) -> Path:
    return config.JOBS_LOG_DIR / f"job_{job_id}.log"


def build_caption(version: CharacterVersion, base_description: str) -> str:
    """Кумулятивний текстовий опис: базовий опис породи + всі
    trait-нотатки по ланцюжку версій (version.prompt_prefix уже
    накопичує їх — див. routers/characters.py). Це і є текстова
    частина "памʼяті" персонажа: вона завжди підмішується в промпт,
    навіть поки LoRA ще недостатньо вивчила нову деталь."""
    return version.prompt_prefix or base_description


def start_training(session: Session, character: Character, version: CharacterVersion) -> Job:
    images_dir = config.CHARACTERS_DIR / str(character.id) / f"v{version.version_number}"
    output_dir = config.LORAS_DIR / str(character.id) / f"v{version.version_number}"
    output_dir.mkdir(parents=True, exist_ok=True)

    job = Job(
        type=JobType.train_lora, status=JobStatus.queued,
        character_id=character.id, character_version_id=version.id,
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    progress_file = _progress_file(job.id)
    log_file = _log_file(job.id)

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--images-dir", str(images_dir),
        "--output-dir", str(output_dir),
        "--trigger-token", character.trigger_token,
        "--caption", build_caption(version, character.base_description),
        "--resolution", str(config.TRAIN_RESOLUTION),
        "--steps", str(config.TRAIN_STEPS),
        "--lr", str(config.TRAIN_LR),
        "--rank", str(config.TRAIN_RANK),
        "--batch-size", str(config.TRAIN_BATCH_SIZE),
        "--progress-file", str(progress_file),
    ]

    with open(log_file, "wb") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(config.BASE_DIR))

    job.pid = proc.pid
    job.status = JobStatus.running
    job.log_path = str(log_file)
    session.add(job)

    version.status = VersionStatus.training
    version.training_job_id = job.id
    session.add(version)

    character.status = CharacterStatus.training
    session.add(character)

    session.commit()
    session.refresh(job)
    return job


def read_progress(job: Job) -> dict:
    if not job.id:
        return {"status": "unknown"}
    pfile = _progress_file(job.id)
    if not pfile.exists():
        return {"status": "queued", "step": 0, "total": 0, "message": "Очікування запуску..."}
    try:
        return json.loads(pfile.read_text())
    except Exception:
        return {"status": "unknown"}
