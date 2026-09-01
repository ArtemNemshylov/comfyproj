"""
Генерація фото для підтвердженого (або ще на перегляді) персонажа:
завжди підмішує його LoRA + накопичений текстовий опис (prompt_prefix),
тому нові фото автоматично "памʼятають" усі підтверджені раніше риси.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from .. import config
from ..comfy_client import ComfyUIClient, ComfyUIError
from ..db import get_session
from ..models import Asset, AssetType, Character, CharacterVersion
from ..workflows import build_photo_workflow, default_positive_prompt

router = APIRouter(prefix="/api/characters", tags=["generate"])


class GeneratePhotoRequest(BaseModel):
    pose_prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    seed: Optional[int] = None


def asset_url(asset: Asset) -> Optional[str]:
    """URL, під яким фронтенд може відкрити файл через змонтовану
    StaticFiles-теку /media/outputs (див. app/main.py)."""
    try:
        rel = Path(asset.file_path).relative_to(config.DATA_DIR / "outputs")
        return f"/media/outputs/{rel.as_posix()}"
    except ValueError:
        return None


def asset_to_dict(asset: Asset) -> dict:
    d = asset.model_dump()
    d["url"] = asset_url(asset)
    return d


def _resolve_version(session: Session, character: Character, version_id: Optional[int]) -> CharacterVersion:
    if version_id:
        version = session.get(CharacterVersion, version_id)
        if not version or version.character_id != character.id:
            raise HTTPException(404, "Версію персонажа не знайдено")
        return version
    if not character.current_version_id:
        raise HTTPException(409, "У персонажа ще немає жодної підтвердженої версії — спершу натренуйте й підтвердіть портрет")
    return session.get(CharacterVersion, character.current_version_id)


def _lora_name_for_comfyui(lora_path: Optional[str]) -> Optional[str]:
    """ComfyUI бачить LoRA-файли через шлях відносно кореня, який описано
    в extra_model_paths.yaml (у нас це admin/data/loras) — тож рахуємо
    шлях відносно config.LORAS_DIR, а не абсолютний шлях на диску."""
    if not lora_path:
        return None
    try:
        return str(Path(lora_path).relative_to(config.LORAS_DIR))
    except ValueError:
        return lora_path  # вже відносний або поза очікуваною текою


def generate_photo_asset(
    session: Session,
    character: Character,
    version: CharacterVersion,
    pose_prompt: str,
    negative_prompt: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    seed: Optional[int] = None,
) -> Asset:
    client = ComfyUIClient()
    if not client.is_alive():
        raise HTTPException(
            503,
            "ComfyUI недоступний на " + config.COMFYUI_URL +
            ". Запустіть його (scripts/run_all.sh у звичайному терміналі macOS) і спробуйте ще раз.",
        )

    positive = default_positive_prompt(version.prompt_prefix, character.trigger_token, pose_prompt)
    negative = negative_prompt or config.DEFAULT_NEGATIVE_PROMPT

    workflow = build_photo_workflow(
        checkpoint=character.base_checkpoint,
        lora_path=_lora_name_for_comfyui(version.lora_path),
        lora_strength=config.DEFAULT_LORA_STRENGTH,
        positive_prompt=positive,
        negative_prompt=negative,
        width=width or config.GEN_WIDTH,
        height=height or config.GEN_HEIGHT,
        steps=steps or config.GEN_STEPS,
        cfg=cfg or config.GEN_CFG,
        sampler=config.GEN_SAMPLER,
        scheduler=config.GEN_SCHEDULER,
        seed=seed,
        filename_prefix=f"char{character.id}_v{version.version_number}",
        upscale_model=config.UPSCALE_MODEL if config.GEN_UPSCALE_ENABLED else None,
        final_scale=config.GEN_FINAL_SCALE,
    )

    try:
        prompt_id = client.queue_prompt(workflow)
        history = client.wait_for_result(prompt_id)
    except ComfyUIError as exc:
        raise HTTPException(502, f"Помилка ComfyUI: {exc}") from exc

    files = client.extract_saved_files(history)
    if not files:
        raise HTTPException(500, "ComfyUI не повернув жодного файлу зображення")

    file_info = files[0]
    content = client.fetch_output_file(
        file_info["filename"], file_info.get("subfolder", ""), file_info.get("type", "output")
    )

    out_dir = config.OUTPUTS_PHOTOS_DIR / str(character.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / file_info["filename"]
    out_path.write_bytes(content)

    asset = Asset(
        character_id=character.id,
        character_version_id=version.id,
        type=AssetType.photo,
        file_path=str(out_path),
        prompt=positive,
        negative_prompt=negative,
        seed=seed,
        meta={"pose_prompt": pose_prompt},
    )
    session.add(asset)
    session.commit()
    session.refresh(asset)
    return asset


@router.post("/{character_id}/generate")
def generate_photo(
    character_id: int,
    body: GeneratePhotoRequest,
    version_id: Optional[int] = None,
    session: Session = Depends(get_session),
):
    character = session.get(Character, character_id)
    if not character:
        raise HTTPException(404, "Персонажа не знайдено")
    version = _resolve_version(session, character, version_id)
    asset = generate_photo_asset(
        session, character, version,
        pose_prompt=body.pose_prompt, negative_prompt=body.negative_prompt,
        width=body.width, height=body.height, steps=body.steps, cfg=body.cfg, seed=body.seed,
    )
    return asset_to_dict(asset)


@router.get("/{character_id}/assets")
def list_assets(character_id: int, asset_type: Optional[str] = None, session: Session = Depends(get_session)):
    from sqlmodel import select
    q = select(Asset).where(Asset.character_id == character_id)
    if asset_type:
        q = q.where(Asset.type == asset_type)
    assets = session.exec(q.order_by(Asset.created_at.desc())).all()
    return [asset_to_dict(a) for a in assets]
