#!/usr/bin/env python3
"""
LoRA-тренування персонажа для Z-Image Turbo через ostris/ai-toolkit
(train_lora.py поруч вміє лише SD1.5-diffusers+peft — Z-Image інша
архітектура: DiT (S3-DiT) + окремий Qwen-текстовий енкодер).

Генерує YAML-конфіг ai-toolkit "на льоту", пише .txt-підписи поруч із
референсними фото (формат датасету ai-toolkit: image.ext + image.txt в
тій самій теці), запускає `python ai-toolkit/run.py <config>.yaml` як
підпроцес і парсить його tqdm-подібний stdout ("450/2000 [01:23<04:56,
...]") у --progress-file — той самий JSON-контракт {"status","step",
"total","message","output_path"}, який уже читає
admin/app/training.py::read_progress, тож бекенд/фронтенд не потребують
змін для іншого рушія тренування.

Запускається як окремий підпроцес з admin/app/training.py.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import traceback
from pathlib import Path

import yaml

AI_TOOLKIT_DIR = Path("/app/ai-toolkit")
AI_TOOLKIT_RUN = AI_TOOLKIT_DIR / "run.py"

# tqdm-подібний рядок прогресу: "...  450/2000 [01:23<04:56, 6.06it/s]"
STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")


def write_progress(path: Path, **kwargs) -> None:
    try:
        path.write_text(json.dumps(kwargs, ensure_ascii=False))
    except Exception:
        pass  # прогрес — best effort, не має валити тренування


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--trigger-token", required=True)
    p.add_argument("--caption", required=True)
    p.add_argument("--resolution", type=int, default=768)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--progress-file", required=True)
    return p.parse_args()


def write_captions(images_dir: Path, full_caption: str) -> None:
    """ai-toolkit очікує image.ext + image.txt в одній теці — пишемо той
    самий кумулятивний caption (тригер-токен + опис персонажа) поруч із
    кожним референсним фото цієї версії."""
    for img in images_dir.iterdir():
        if img.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            img.with_suffix(".txt").write_text(full_caption, encoding="utf-8")


def build_config(*, job_name: str, images_dir: Path, output_dir: Path,
                  full_caption: str, resolution: int, steps: int,
                  lr: float, rank: int, batch_size: int) -> dict:
    return {
        "job": "extension",
        "config": {
            "name": job_name,
            "process": [{
                "type": "sd_trainer",
                "training_folder": str(output_dir),
                "device": "cuda:0",
                "network": {"type": "lora", "linear": rank, "linear_alpha": rank},
                "save": {"dtype": "float16", "save_every": steps, "max_step_saves_to_keep": 1},
                "datasets": [{
                    "folder_path": str(images_dir),
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.0,
                    "shuffle_tokens": False,
                    "cache_latents_to_disk": True,
                    "resolution": [resolution],
                }],
                "train": {
                    "batch_size": batch_size,
                    "cache_text_embeddings": True,
                    "steps": steps,
                    "gradient_accumulation": 1,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch",
                    "optimizer": "adamw8bit",
                    "lr": lr,
                    "dtype": "bf16",
                },
                "model": {
                    "name_or_path": "Tongyi-MAI/Z-Image-Turbo",
                    "arch": "zimage",
                    # low_vram=True постійно перекидав частини моделі між CPU/GPU —
                    # VRAM однаково впритул забита (12/12ГБ) і без цього режиму,
                    # тобто вигоди немає, лише штраф швидкості (~41с/крок,
                    # 15-20x повільніше за community-орієнтир 2-3с/крок).
                    "low_vram": False,
                },
                # sample_every > steps -> без прев'ю під час тренування
                # (економія часу й VRAM для POC; прев'ю все одно доступне
                # через звичайну генерацію після тренування)
                "sample": {
                    "sampler": "flowmatch",
                    "sample_every": steps + 1,
                    "width": resolution,
                    "height": resolution,
                    "prompts": [full_caption],
                    "neg": "",
                    "seed": 42,
                    "walk_seed": False,
                    "guidance_scale": 1,
                    "sample_steps": 8,
                },
            }],
        },
        "meta": {"name": job_name, "version": "1.0"},
    }


def find_output_lora(output_dir: Path, job_name: str) -> "Path | None":
    job_dir = output_dir / job_name
    candidates = sorted(job_dir.glob("*.safetensors"), key=lambda p: p.stat().st_mtime) if job_dir.exists() else []
    return candidates[-1] if candidates else None


def main() -> int:
    args = parse_args()
    progress_path = Path(args.progress_file)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    write_progress(progress_path, status="starting", step=0, total=args.steps, message="Підготовка конфігурації ai-toolkit...")

    try:
        images_dir = Path(args.images_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        full_caption = f"{args.trigger_token}, {args.caption}".strip(", ")
        write_captions(images_dir, full_caption)

        job_name = f"char_{images_dir.parent.name}_{images_dir.name}"
        cfg = build_config(
            job_name=job_name, images_dir=images_dir, output_dir=output_dir,
            full_caption=full_caption, resolution=args.resolution, steps=args.steps,
            lr=args.lr, rank=args.rank, batch_size=args.batch_size,
        )
        config_path = output_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        write_progress(progress_path, status="running", step=0, total=args.steps, message="Запуск ai-toolkit...")

        proc = subprocess.Popen(
            [sys.executable, str(AI_TOOLKIT_RUN), str(config_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            cwd=str(AI_TOOLKIT_DIR),
        )
        tail: list[str] = []
        for line in proc.stdout:
            # ai-toolkit оновлює tqdm через \r, тож ітерація по рядках (\n)
            # часто віддає лише один "фінальний" рядок на кожен прогрес-бар —
            # включно з проміжними барами препроцесингу (кешування латентів
            # "1/1" чи "4/4" для наших кількох референсних фото), а не лише
            # для самого циклу тренування. Приймаємо для прогрес-файлу ЛИШЕ
            # рядки, де total збігається з реальною ціллю кроків (args.steps)
            # — інакше проміжний бар преробки помилково показував "100%".
            tail.append(line)
            tail[:] = tail[-50:]
            m = STEP_RE.search(line)
            if m and int(m.group(2)) == args.steps:
                step, total = int(m.group(1)), int(m.group(2))
                write_progress(progress_path, status="running", step=step, total=total, message=line.strip())
        ret = proc.wait()

        if ret != 0:
            raise RuntimeError(f"ai-toolkit завершився з кодом {ret}. Останні рядки логу:\n" + "".join(tail))

        write_progress(progress_path, status="saving", step=args.steps, total=args.steps, message="Пошук збереженого файлу LoRA...")
        lora_file = find_output_lora(output_dir, job_name)
        if not lora_file:
            raise RuntimeError(f"ai-toolkit завершився успішно, але .safetensors не знайдено в {output_dir / job_name}")

        write_progress(
            progress_path, status="done", step=args.steps, total=args.steps,
            message="Готово", output_path=str(lora_file),
        )
        return 0

    except Exception as exc:
        write_progress(
            progress_path, status="error", step=0, total=args.steps,
            message=str(exc), traceback=traceback.format_exc(),
        )
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
