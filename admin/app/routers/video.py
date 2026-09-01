"""
Генерація короткого відео з персонажем (image-to-video через AnimateDiff
у ComfyUI) — той самий checkpoint+LoRA стек, що й для фото, тому пес
лишається впізнаваним і на відео.

POC-попередження: на MacBook без dedicated GPU (навіть на MPS) відео —
найважча операція тут. Дефолтні параметри (config.VIDEO_*) навмисно
маленькі (384x384, 16 кадрів). Перший запуск може тривати кілька
хвилин — це очікувано.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from .. import config
from ..comfy_client import ComfyUIClient, ComfyUIError
from ..db import get_session
from ..models import Asset, AssetType, Character, CharacterVersion
from ..workflows import build_video_workflow, default_positive_prompt
from .generate import _resolve_version, _lora_name_for_comfyui, asset_to_dict

router = APIRouter(prefix="/api/characters", tags=["video"])


class GenerateVideoRequest(BaseModel):
    motion_prompt: str
    negative_prompt: Optional[str] = None
    frames: Optional[int] = None
    fps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    seed: Optional[int] = None


@router.post("/{character_id}/generate-video")
def generate_video(
    character_id: int,
    body: GenerateVideoRequest,
    version_id: Optional[int] = None,
    session: Session = Depends(get_session),
):
    character = session.get(Character, character_id)
    if not character:
        raise HTTPException(404, "Персонажа не знайдено")
    version = _resolve_version(session, character, version_id)

    client = ComfyUIClient()
    if not client.is_alive():
        raise HTTPException(
            503,
            "ComfyUI недоступний на " + config.COMFYUI_URL +
            ". Запустіть його (scripts/run_all.sh у звичайному терміналі macOS) і спробуйте ще раз.",
        )

    # У Docker-розгортанні admin-контейнер зазвичай не монтує цю теку
    # (моделі AnimateDiff потрібні лише comfyui-контейнеру) — тоді просто
    # пропускаємо цю "приємну" перевірку і даємо ComfyUI самому сказати,
    # якщо файлу справді немає. Валимо запит заздалегідь лише тоді, коли
    # тека взагалі видима (нативний запуск на macOS) і файлу в ній нема.
    motion_module_path = config.ANIMATEDIFF_MODELS_DIR / config.ANIMATEDIFF_MOTION_MODULE
    if config.ANIMATEDIFF_MODELS_DIR.exists() and not motion_module_path.exists():
        raise HTTPException(
            409,
            f"Motion-модуль {config.ANIMATEDIFF_MOTION_MODULE} не знайдено у {motion_module_path.parent}. "
            f"Довантажте його (scripts/setup_comfyui.sh, або docker/entrypoint-comfyui.sh при Docker-запуску).",
        )

    positive = default_positive_prompt(version.prompt_prefix, character.trigger_token, body.motion_prompt)
    negative = body.negative_prompt or config.DEFAULT_NEGATIVE_PROMPT

    workflow = build_video_workflow(
        checkpoint=character.base_checkpoint,
        lora_path=_lora_name_for_comfyui(version.lora_path),
        lora_strength=config.DEFAULT_LORA_STRENGTH,
        motion_module=config.ANIMATEDIFF_MOTION_MODULE,
        positive_prompt=positive,
        negative_prompt=negative,
        width=body.width or config.VIDEO_WIDTH,
        height=body.height or config.VIDEO_HEIGHT,
        frames=body.frames or config.VIDEO_FRAMES,
        fps=body.fps or config.VIDEO_FPS,
        steps=body.steps or config.VIDEO_STEPS,
        cfg=body.cfg or config.VIDEO_CFG,
        sampler=config.GEN_SAMPLER,
        scheduler=config.GEN_SCHEDULER,
        seed=body.seed,
        filename_prefix=f"char{character.id}_v{version.version_number}_video",
    )

    try:
        prompt_id = client.queue_prompt(workflow)
        # Відео рахується довше за фото — даємо значно більший таймаут.
        history = client.wait_for_result(prompt_id, poll_interval=3.0, max_wait=3600.0)
    except ComfyUIError as exc:
        raise HTTPException(
            502,
            f"Помилка ComfyUI під час генерації відео: {exc}. Якщо це "
            f"'Unknown node type' — перевірте app/workflows.py::NODE_TYPES "
            f"проти /object_info вашої версії ComfyUI-AnimateDiff-Evolved.",
        ) from exc

    files = client.extract_saved_files(history)
    if not files:
        raise HTTPException(500, "ComfyUI не повернув жодного відеофайлу")

    file_info = files[0]
    content = client.fetch_output_file(
        file_info["filename"], file_info.get("subfolder", ""), file_info.get("type", "output")
    )

    out_dir = config.OUTPUTS_VIDEOS_DIR / str(character.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / file_info["filename"]
    out_path.write_bytes(content)

    asset = Asset(
        character_id=character.id,
        character_version_id=version.id,
        type=AssetType.video,
        file_path=str(out_path),
        prompt=positive,
        negative_prompt=negative,
        seed=body.seed,
        meta={"motion_prompt": body.motion_prompt, "frames": body.frames or config.VIDEO_FRAMES},
    )
    session.add(asset)
    session.commit()
    session.refresh(asset)
    return asset_to_dict(asset)
