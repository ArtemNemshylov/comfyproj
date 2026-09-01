#!/usr/bin/env python3
"""
Мінімальний DreamBooth-LoRA тренер для SD1.5 — навмисно "бідної якості" /
POC-режим: низька роздільність, мало кроків, маленький rank. Мета —
щоб MacBook (M-серія, MPS) не перегрівався і тренування одного
персонажа займало хвилини, а не години. Якщо треба якість — підніміть
--resolution/--steps/--rank (і озбройтесь терпінням).

Запускається як окремий підпроцес з admin/app/training.py:
    python3 train_lora.py --images-dir ... --output-dir ...
        --trigger-token sks_rex --caption "..." --progress-file ...

Прогрес пишеться у --progress-file як JSON: {"status", "step", "total",
"message"} — це читає бекенд, поки підпроцес крутиться.

Формат збереження: LoRA конвертується у kohya-сумісний вигляд
(convert_state_dict_to_kohya), бо саме такий формат очікує нода
LoraLoader у ComfyUI. Якщо після реального тренування ComfyUI все ж
не зможе завантажити файл — перевірте назви ключів у safetensors
(safetensors.safe_open) і версію diffusers (потрібен >=0.27).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path


def write_progress(path: Path, **kwargs) -> None:
    try:
        path.write_text(json.dumps(kwargs, ensure_ascii=False))
    except Exception:
        pass  # прогрес — best effort, не має валити тренування


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True, help="Тека з референсними фото (кумулятивні для цієї версії)")
    p.add_argument("--output-dir", required=True, help="Куди зберегти lora.safetensors")
    p.add_argument("--base-model", required=True, help="Шлях до .safetensors базового чекпоінта (SD1.5)")
    p.add_argument("--trigger-token", required=True, help='Унікальний токен, напр. "sks_rex"')
    p.add_argument("--caption", required=True, help="Базовий опис + накопичені trait-нотатки, одним рядком")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress-file", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    progress_path = Path(args.progress_file)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    write_progress(progress_path, status="starting", step=0, total=args.steps, message="Завантаження бібліотек...")

    try:
        import torch
        from PIL import Image
        from torchvision import transforms
        from diffusers import StableDiffusionPipeline, DDPMScheduler
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict
        from safetensors.torch import save_file

        try:
            from diffusers.utils.state_dict_utils import convert_state_dict_to_kohya
        except ImportError:  # старіші/новіші diffusers можуть тримати це у іншому місці
            from diffusers.utils import convert_state_dict_to_kohya  # type: ignore

        random.seed(args.seed)
        torch.manual_seed(args.seed)

        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        write_progress(progress_path, status="loading_model", step=0, total=args.steps, message=f"Завантаження {args.base_model} на {device}...")

        weight_dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
        pipe = StableDiffusionPipeline.from_single_file(
            args.base_model, torch_dtype=weight_dtype, safety_checker=None,
        )
        pipe.to(device)

        unet = pipe.unet
        vae = pipe.vae
        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer
        noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)

        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            init_lora_weights="gaussian",
        )
        unet.add_adapter(lora_config)
        unet.to(device)

        trainable_params = [p for p in unet.parameters() if p.requires_grad]
        write_progress(progress_path, status="preparing_data", step=0, total=args.steps, message="Підготовка датасету...")

        images_dir = Path(args.images_dir)
        image_paths = sorted(
            [p for p in images_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
        )
        if not image_paths:
            raise RuntimeError(f"У {images_dir} немає жодного зображення (.png/.jpg/.webp)")

        tfm = transforms.Compose([
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        caption = f"a photo of {args.trigger_token}, {args.caption}".strip(", ")
        tok = tokenizer(caption, padding="max_length", truncation=True,
                        max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            encoder_hidden_states = text_encoder(tok)[0].to(weight_dtype)

        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

        write_progress(progress_path, status="running", step=0, total=args.steps, message="Тренування...")

        unet.train()
        for step in range(args.steps):
            img_path = image_paths[step % len(image_paths)]
            image = Image.open(img_path).convert("RGB")
            pixel_values = tfm(image).unsqueeze(0).to(device=device, dtype=weight_dtype)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0 or step == args.steps - 1:
                write_progress(
                    progress_path, status="running", step=step + 1, total=args.steps,
                    message=f"Крок {step + 1}/{args.steps}, loss={float(loss):.4f}",
                )

        write_progress(progress_path, status="saving", step=args.steps, total=args.steps, message="Збереження LoRA...")

        peft_state_dict = get_peft_model_state_dict(unet)
        kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "lora.safetensors"
        save_file(kohya_state_dict, str(output_path))

        write_progress(
            progress_path, status="done", step=args.steps, total=args.steps,
            message="Готово", output_path=str(output_path),
        )
        return 0

    except Exception as exc:  # POC: краще впасти з чітким повідомленням, ніж мовчки
        write_progress(
            progress_path, status="error", step=0, total=args.steps,
            message=str(exc), traceback=traceback.format_exc(),
        )
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
