"""
CRUD персонажів (собак) + весь цикл "памʼяті": створення версії,
тренування LoRA, перегляд прев'ю, підтвердження/відхилення, додавання
нової риси (нова версія, що успадковує все попереднє).
"""
from __future__ import annotations

import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from sqlmodel import Session, select

from .. import config, training
from ..db import get_session
from ..models import (
    Character, CharacterVersion, Job,
    CharacterStatus, VersionStatus, JobStatus,
)

router = APIRouter(prefix="/api/characters", tags=["characters"])


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "dog"


def _save_uploads(files: List[UploadFile], target_dir: Path) -> List[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        suffix = Path(f.filename or "").suffix or ".png"
        dest = target_dir / f"{uuid.uuid4().hex}{suffix}"
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(str(dest))
    return saved


def _get_character_or_404(session: Session, character_id: int) -> Character:
    char = session.get(Character, character_id)
    if not char:
        raise HTTPException(404, "Персонажа (собаку) не знайдено")
    return char


def _get_version_or_404(session: Session, character_id: int, version_id: int) -> CharacterVersion:
    version = session.get(CharacterVersion, version_id)
    if not version or version.character_id != character_id:
        raise HTTPException(404, "Версію персонажа не знайдено")
    return version


def _run_preview_generation(character_id: int, version_id: int) -> None:
    """Викликається у фоні одразу після успішного тренування: генерує
    декілька типових поз, щоб було що показати користувачу на екрані
    підтвердження портрета."""
    from ..db import engine
    from sqlmodel import Session as _Session
    from .generate import generate_photo_asset

    with _Session(engine) as session:
        character = session.get(Character, character_id)
        version = session.get(CharacterVersion, version_id)
        if not character or not version:
            return
        for pose in config.PREVIEW_POSES_AFTER_TRAINING:
            try:
                generate_photo_asset(session, character, version, pose_prompt=pose)
            except Exception:
                continue  # прев'ю — best effort, не валимо фіналізацію тренування через це


@router.get("")
def list_characters(session: Session = Depends(get_session)):
    return session.exec(select(Character)).all()


@router.post("")
def create_character(
    name: str = Form(...),
    base_description: str = Form(""),
    files: List[UploadFile] = File(...),
    session: Session = Depends(get_session),
):
    """Створення нового персонажа з "першою фотографією" (чи кількома).
    trigger_token генерується автоматично й лишається незмінним на все
    життя персонажа — саме він завжди підмішується у промпт, щоб модель
    "впізнавала" цю конкретну собаку."""
    if not files:
        raise HTTPException(400, "Потрібно завантажити хоча б одне референсне фото")

    trigger_token = f"sks_{_slugify(name)}"
    character = Character(
        name=name,
        base_description=base_description,
        trigger_token=trigger_token,
        base_checkpoint=config.BASE_CHECKPOINT,
        status=CharacterStatus.draft,
    )
    session.add(character)
    session.commit()
    session.refresh(character)

    version = CharacterVersion(
        character_id=character.id,
        parent_version_id=None,
        version_number=1,
        trait_note="Початковий портрет",
        prompt_prefix=base_description,
        status=VersionStatus.pending,
    )
    session.add(version)
    session.commit()
    session.refresh(version)

    images_dir = config.CHARACTERS_DIR / str(character.id) / f"v{version.version_number}"
    saved_paths = _save_uploads(files, images_dir)
    version.reference_images = saved_paths
    session.add(version)
    session.commit()
    session.refresh(character)
    return character


def _version_to_dict(version: CharacterVersion) -> dict:
    d = version.model_dump()
    urls = []
    for path in version.reference_images or []:
        try:
            rel = Path(path).relative_to(config.CHARACTERS_DIR)
            urls.append(f"/media/characters/{rel.as_posix()}")
        except ValueError:
            continue
    d["reference_image_urls"] = urls
    return d


@router.get("/{character_id}")
def get_character(character_id: int, session: Session = Depends(get_session)):
    character = _get_character_or_404(session, character_id)
    versions = session.exec(
        select(CharacterVersion)
        .where(CharacterVersion.character_id == character_id)
        .order_by(CharacterVersion.version_number)
    ).all()
    return {"character": character, "versions": [_version_to_dict(v) for v in versions]}


@router.post("/{character_id}/versions/{version_id}/train")
def train_version(character_id: int, version_id: int, session: Session = Depends(get_session)):
    character = _get_character_or_404(session, character_id)
    version = _get_version_or_404(session, character_id, version_id)
    if version.status == VersionStatus.training:
        raise HTTPException(409, "Ця версія вже тренується")
    if not version.reference_images:
        raise HTTPException(400, "У версії немає референсних фото для тренування")

    checkpoint_path = config.CHECKPOINTS_DIR / character.base_checkpoint
    if not checkpoint_path.exists():
        raise HTTPException(
            409,
            f"Базовий чекпоінт {character.base_checkpoint} не знайдено у "
            f"{config.CHECKPOINTS_DIR}. Нативний запуск: спершу виконайте "
            f"scripts/setup_comfyui.sh у звичайному терміналі macOS. "
            f"Docker-запуск: дочекайтесь, поки comfyui-контейнер довантажить "
            f"моделі (docker/entrypoint-comfyui.sh) — дивіться docker compose logs comfyui.",
        )

    job = training.start_training(session, character, version)
    return {"job_id": job.id, "version_id": version.id, "status": version.status}


@router.get("/{character_id}/versions/{version_id}/training-status")
def training_status(
    character_id: int, version_id: int,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    character = _get_character_or_404(session, character_id)
    version = _get_version_or_404(session, character_id, version_id)
    if not version.training_job_id:
        return {"status": "not_started"}

    job = session.get(Job, version.training_job_id)
    progress = training.read_progress(job)

    if progress.get("status") == "done" and version.status != VersionStatus.ready_for_review:
        lora_path = progress.get("output_path") or str(
            config.LORAS_DIR / str(character.id) / f"v{version.version_number}" / "lora.safetensors"
        )
        version.status = VersionStatus.ready_for_review
        version.lora_path = lora_path
        session.add(version)

        job.status = JobStatus.succeeded
        job.finished_at = datetime.utcnow()
        session.add(job)
        session.commit()

        background_tasks.add_task(_run_preview_generation, character.id, version.id)

    elif progress.get("status") == "error" and version.status != VersionStatus.failed:
        version.status = VersionStatus.failed
        session.add(version)
        job.status = JobStatus.failed
        job.error = progress.get("message")
        job.finished_at = datetime.utcnow()
        session.add(job)
        session.commit()

    return {"status": version.status, "progress": progress}


@router.post("/{character_id}/versions/{version_id}/confirm")
def confirm_version(character_id: int, version_id: int, session: Session = Depends(get_session)):
    character = _get_character_or_404(session, character_id)
    version = _get_version_or_404(session, character_id, version_id)
    if version.status != VersionStatus.ready_for_review:
        raise HTTPException(409, "Підтвердити можна лише версію зі статусом 'ready_for_review'")

    version.status = VersionStatus.confirmed
    version.confirmed_at = datetime.utcnow()
    session.add(version)

    character.current_version_id = version.id
    character.status = CharacterStatus.confirmed
    session.add(character)

    session.commit()
    session.refresh(character)
    return character


@router.post("/{character_id}/versions/{version_id}/reject")
def reject_version(character_id: int, version_id: int, session: Session = Depends(get_session)):
    version = _get_version_or_404(session, character_id, version_id)
    version.status = VersionStatus.rejected
    session.add(version)
    session.commit()
    return version


@router.post("/{character_id}/new-version")
def new_version(
    character_id: int,
    trait_note: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    session: Session = Depends(get_session),
):
    """Додає нову рису персонажу (напр. "зʼявився шрам на лівому вусі") і
    створює НОВУ версію, що успадковує референсні фото й накопичений
    текстовий опис попередньої підтвердженої версії — це і є "памʼять":
    все, що було підтверджено раніше, автоматично тягнеться у майбутнє."""
    character = _get_character_or_404(session, character_id)
    if not character.current_version_id:
        raise HTTPException(409, "У персонажа ще немає підтвердженої версії — спершу підтвердіть початковий портрет")

    parent = session.get(CharacterVersion, character.current_version_id)
    new_version_number = parent.version_number + 1

    new_prefix = f"{parent.prompt_prefix}, {trait_note}".strip(", ")
    new_version = CharacterVersion(
        character_id=character.id,
        parent_version_id=parent.id,
        version_number=new_version_number,
        trait_note=trait_note,
        prompt_prefix=new_prefix,
        status=VersionStatus.pending,
    )
    session.add(new_version)
    session.commit()
    session.refresh(new_version)

    images_dir = config.CHARACTERS_DIR / str(character.id) / f"v{new_version_number}"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Кумулятивно копіюємо референсні фото попередньої версії...
    copied_paths = []
    for old_path in parent.reference_images:
        old_path_obj = Path(old_path)
        if old_path_obj.exists():
            dest = images_dir / old_path_obj.name
            shutil.copyfile(old_path_obj, dest)
            copied_paths.append(str(dest))

    # ...і додаємо нові фото (де видно нову рису, напр. шрам).
    new_paths = _save_uploads(files, images_dir) if files else []

    new_version.reference_images = copied_paths + new_paths
    session.add(new_version)
    session.commit()
    session.refresh(new_version)
    return new_version
