"""
Побудова ComfyUI workflow-графів (API-формат: dict {node_id: {class_type, inputs}}).

Дві функції:
  - build_photo_workflow(...)  — базові ноди ComfyUI (стабільні у всіх версіях)
  - build_video_workflow(...)  — базові ноди + AnimateDiff-Evolved + VideoHelperSuite

Важливо про build_video_workflow: назви класів кастомних нод
AnimateDiff-Evolved (ADE_*) періодично змінюються між версіями репо.
Якщо ComfyUI поверне ComfyUIError з node_errors — відкрийте
http://127.0.0.1:8188/object_info (або ComfyUIClient.object_info())
і знайдіть точні class_type у встановленої версії, потім поправте
константи в NODE_TYPES нижче. Це єдине місце, де це треба робити.
"""
from __future__ import annotations

import random
from typing import Optional

from . import config

# Точки, які найімовірніше доведеться підправити після оновлення
# ComfyUI-AnimateDiff-Evolved / ComfyUI-VideoHelperSuite.
NODE_TYPES = {
    "animatediff_loader": "ADE_LoadAnimateDiffModel",
    "animatediff_apply": "ADE_ApplyAnimateDiffModelSimple",
    "animatediff_sampling": "ADE_UseEvolvedSampling",
    "video_combine": "VHS_VideoCombine",
}


def _seed(seed: Optional[int]) -> int:
    return seed if seed is not None else random.randint(0, 2**31 - 1)


def build_photo_workflow(
    *,
    checkpoint: str,
    lora_path: Optional[str],
    lora_strength: float,
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: Optional[int],
    filename_prefix: str,
    upscale_model: Optional[str] = None,
    final_scale: float = 1.0,
) -> dict:
    """Стандартний txt2img-граф: Checkpoint -> (LoRA) -> CLIP -> KSampler -> VAEDecode
    -> (ESRGAN-апскейл) -> SaveImage. Використовує лише "рідні" ноди ComfyUI
    (+ вбудовану підтримку upscale-моделей), тому стабільний між версіями.

    Роздільність підіймається апскейлом ПІСЛЯ генерації (upscale_model), а
    не збільшенням width/height у латенті — SD1.5, згенероване напряму
    вище ~576px, часто ламає анатомію ("дві голови"). Якщо upscale_model
    не задано — апскейл пропускається, повертається звичайний VAEDecode."""
    g: dict = {}

    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}}

    model_link = ["1", 0]
    clip_link = ["1", 1]

    if lora_path:
        g["2"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_link,
                "clip": clip_link,
                "lora_name": lora_path,
                "strength_model": lora_strength,
                "strength_clip": lora_strength,
            },
        }
        model_link = ["2", 0]
        clip_link = ["2", 1]

    g["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": positive_prompt, "clip": clip_link}}
    g["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": clip_link}}
    g["5"] = {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
    g["6"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_link,
            "positive": ["3", 0],
            "negative": ["4", 0],
            "latent_image": ["5", 0],
            "seed": _seed(seed),
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise": 1.0,
        },
    }
    g["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}}

    final_image_link = ["7", 0]
    if upscale_model:
        g["30"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": upscale_model}}
        g["31"] = {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {"upscale_model": ["30", 0], "image": final_image_link},
        }
        final_image_link = ["31", 0]
        if final_scale and final_scale > 0:
            g["32"] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image": final_image_link,
                    "upscale_method": "lanczos",
                    "width": int(width * final_scale),
                    "height": int(height * final_scale),
                    "crop": "disabled",
                },
            }
            final_image_link = ["32", 0]

    g["8"] = {
        "class_type": "SaveImage",
        "inputs": {"images": final_image_link, "filename_prefix": filename_prefix},
    }
    return g


def build_video_workflow(
    *,
    checkpoint: str,
    lora_path: Optional[str],
    lora_strength: float,
    motion_module: str,
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    frames: int,
    fps: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: Optional[int],
    filename_prefix: str,
) -> dict:
    """AnimateDiff-граф: той самий checkpoint+LoRA стек, що й для фото
    (тому персонаж лишається впізнаваним і у відео), плюс AnimateDiff
    motion-модуль, латент-батч розміром `frames` та комбінування кадрів
    у відеофайл через VHS_VideoCombine."""
    g: dict = {}

    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}}
    model_link = ["1", 0]
    clip_link = ["1", 1]

    if lora_path:
        g["2"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_link,
                "clip": clip_link,
                "lora_name": lora_path,
                "strength_model": lora_strength,
                "strength_clip": lora_strength,
            },
        }
        model_link = ["2", 0]
        clip_link = ["2", 1]

    g["10"] = {
        "class_type": NODE_TYPES["animatediff_loader"],
        "inputs": {"model_name": motion_module},
    }
    g["11"] = {
        "class_type": NODE_TYPES["animatediff_apply"],
        "inputs": {"motion_model": ["10", 0]},
    }
    g["12"] = {
        "class_type": NODE_TYPES["animatediff_sampling"],
        "inputs": {"model": model_link, "m_models": ["11", 0], "context_options": None},
    }
    animated_model_link = ["12", 0]

    g["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": positive_prompt, "clip": clip_link}}
    g["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": clip_link}}
    g["5"] = {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": frames}}
    g["6"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": animated_model_link,
            "positive": ["3", 0],
            "negative": ["4", 0],
            "latent_image": ["5", 0],
            "seed": _seed(seed),
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise": 1.0,
        },
    }
    g["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}}
    g["9"] = {
        "class_type": NODE_TYPES["video_combine"],
        "inputs": {
            "images": ["7", 0],
            "frame_rate": fps,
            "filename_prefix": filename_prefix,
            "format": "video/h264-mp4",
            "save_output": True,
        },
    }
    return g


def build_zimage_workflow(
    *,
    unet_name: str,
    clip_name: str,
    vae_name: str,
    positive_prompt: str,
    width: int,
    height: int,
    seed: Optional[int],
    filename_prefix: str,
    steps: int = 8,
    shift: float = 3,
    lora_path: Optional[str] = None,
    lora_strength: float = 1.0,
) -> dict:
    """Z-Image-Turbo (Alibaba Tongyi Lab, S3-DiT, 6B) — архітектура повністю
    інша за SD1.5/SDXL: окремі UNETLoader/CLIPLoader(type=lumina2)/VAELoader
    замість одного CheckpointLoaderSimple, EmptySD3LatentImage замість
    EmptyLatentImage, і ModelSamplingAuraFlow перед KSampler. Це "turbo"
    (дистильована) модель — cfg=1 і немає реального негативного промпту,
    негативна кондиція просто зануляється (ConditioningZeroOut). Точний граф
    узятий з офіційного ComfyUI-шаблону "image_z_image_turbo_int8".
    Sampler/scheduler (res_multistep/simple) і steps=8/shift=3 — дефолти
    того ж шаблону, під int8-модель."""
    g: dict = {}

    g["28"] = {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}}
    g["30"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "lumina2", "device": "default"}}
    g["29"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}}

    model_link = ["28", 0]
    if lora_path:
        g["40"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model_link, "lora_name": lora_path, "strength_model": lora_strength},
        }
        model_link = ["40", 0]

    g["27"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["30", 0], "text": positive_prompt},
    }
    g["33"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["27", 0]}}
    g["13"] = {
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
    g["11"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model_link, "shift": shift}}
    g["3"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["11", 0],
            "positive": ["27", 0],
            "negative": ["33", 0],
            "latent_image": ["13", 0],
            "seed": _seed(seed),
            "steps": steps,
            "cfg": 1,
            "sampler_name": "res_multistep",
            "scheduler": "simple",
            "denoise": 1,
        },
    }
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["29", 0]}}
    g["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": filename_prefix}}
    return g


def default_positive_prompt(prompt_prefix: str, trigger_token: str, pose_or_motion: str) -> str:
    parts = [p.strip() for p in (trigger_token, prompt_prefix, pose_or_motion) if p and p.strip()]
    return ", ".join(parts)
