"""
Довільна генерація фото за промптом, без прив'язки до персонажа/LoRA —
просто базовий чекпоінт (config.BASE_CHECKPOINT) + текстовий промпт.
Нічого не пише в БД (Asset вимагає character_id/character_version_id),
файли лежать у DATA_DIR/outputs/freeform і роздаються через уже
змонтовану /media/outputs StaticFiles-теку (app/main.py).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import config
from ..comfy_client import ComfyUIClient, ComfyUIError
from ..workflows import build_photo_workflow, build_zimage_workflow

router = APIRouter(prefix="/api/generate/freeform", tags=["freeform"])

FREEFORM_DIR = config.DATA_DIR / "outputs" / "freeform"
FREEFORM_DIR.mkdir(parents=True, exist_ok=True)


class FreeformRequest(BaseModel):
    prompt: str
    engine: str = "sd15"  # "sd15" (Realistic Vision) | "zimage" (Z-Image Turbo)
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    seed: Optional[int] = None


def _asset_url(path: Path) -> str:
    rel = path.relative_to(config.DATA_DIR / "outputs")
    return f"/media/outputs/{rel.as_posix()}"


@router.post("")
def generate_freeform(body: FreeformRequest):
    if not body.prompt.strip():
        raise HTTPException(400, "Промпт не може бути порожнім")

    client = ComfyUIClient()
    if not client.is_alive():
        raise HTTPException(
            503,
            "ComfyUI недоступний на " + config.COMFYUI_URL + ". Запустіть його і спробуйте ще раз.",
        )

    # Резолвимо seed тут (а не всередині build_*_workflow), щоб мати змогу
    # повернути реально використане значення — без цього при seed=null
    # ComfyUI сам обирає випадкове число, і користувач ніколи не дізнається
    # яке саме (а без цього не можна повторити той самий "образ" персонажа
    # для консистентності на кількох кадрах).
    seed = body.seed if body.seed is not None else random.randint(0, 2**31 - 1)

    if body.engine == "zimage":
        workflow = build_zimage_workflow(
            unet_name=config.ZIMAGE_UNET,
            clip_name=config.ZIMAGE_CLIP,
            vae_name=config.ZIMAGE_VAE,
            positive_prompt=body.prompt.strip(),
            width=body.width or 1024,
            height=body.height or 1024,
            seed=seed,
            filename_prefix="freeform_zimage",
        )
    else:
        workflow = build_photo_workflow(
            checkpoint=config.BASE_CHECKPOINT,
            lora_path=None,
            lora_strength=0.0,
            positive_prompt=body.prompt.strip(),
            negative_prompt=body.negative_prompt or config.DEFAULT_NEGATIVE_PROMPT,
            width=body.width or config.GEN_WIDTH,
            height=body.height or config.GEN_HEIGHT,
            steps=body.steps or config.GEN_STEPS,
            cfg=body.cfg or config.GEN_CFG,
            sampler=config.GEN_SAMPLER,
            scheduler=config.GEN_SCHEDULER,
            seed=seed,
            filename_prefix="freeform",
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

    out_path = FREEFORM_DIR / file_info["filename"]
    out_path.write_bytes(content)

    return {
        "url": _asset_url(out_path),
        "prompt": body.prompt.strip(),
        "negative_prompt": body.negative_prompt or config.DEFAULT_NEGATIVE_PROMPT,
        "seed": seed,
    }


@router.get("/history")
def freeform_history(limit: int = 30):
    files = sorted(FREEFORM_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"url": _asset_url(p)} for p in files[:limit]]
