"""
Конфігурація адмінки compfui.

Усі шляхи до моделей/даних беруться з env-змінних COMPF_* з розумними
дефолтами під нативний запуск на macOS (comfyui/ як сусідня тека). Для
Docker-розгортання (docker-compose.yml у корені) ці змінні виставляються
інакше — контейнери бачать спільні томи по інших шляхах. Це дозволяє
одному й тому ж коду однаково працювати і "руками" на маку, і в Docker
на десктопі з NVIDIA GPU.

Параметри генерації/тренування нижче виставлені під дискретну GPU
(перевірено на RTX 3070 12ГБ) — для слабшого заліза (MacBook) знижуйте
через ті самі COMPF_*-змінні.
"""
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent  # .../admin

DATA_DIR = Path(os.environ.get("COMPF_DATA_DIR", str(BASE_DIR / "data")))
CHARACTERS_DIR = DATA_DIR / "characters"
LORAS_DIR = DATA_DIR / "loras"
OUTPUTS_PHOTOS_DIR = DATA_DIR / "outputs" / "photos"
OUTPUTS_VIDEOS_DIR = DATA_DIR / "outputs" / "videos"
JOBS_LOG_DIR = DATA_DIR / "jobs"
DB_PATH = DATA_DIR / "compf.db"

for _d in (CHARACTERS_DIR, LORAS_DIR, OUTPUTS_PHOTOS_DIR, OUTPUTS_VIDEOS_DIR, JOBS_LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- ComfyUI ---
# Нативно: 127.0.0.1. У docker-compose: COMFYUI_HOST=comfyui (імʼя сервісу
# в внутрішній мережі compose), див. docker-compose.yml.
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

BASE_CHECKPOINT = os.environ.get("COMPF_BASE_CHECKPOINT", "v1-5-pruned-emaonly.safetensors")
ANIMATEDIFF_MOTION_MODULE = os.environ.get("COMPF_MOTION_MODULE", "mm_sd_v15_v2.ckpt")

# Шлях до реального клону ComfyUI поруч із admin/ (нативний запуск,
# scripts/setup_comfyui.sh). У Docker admin-контейнер не має доступу до
# файлової системи comfyui-контейнера напряму — там ці три шляхи
# перевизначаються env-змінними на примонтовані спільні томи
# (COMPF_CHECKPOINTS_DIR=/models/checkpoints тощо, див. docker-compose.yml).
COMFYUI_DIR = Path(os.environ.get("COMPF_COMFYUI_DIR", str(BASE_DIR.parent / "comfyui")))
CHECKPOINTS_DIR = Path(os.environ.get(
    "COMPF_CHECKPOINTS_DIR", str(COMFYUI_DIR / "models" / "checkpoints"),
))
ANIMATEDIFF_MODELS_DIR = Path(os.environ.get(
    "COMPF_ANIMATEDIFF_MODELS_DIR",
    str(COMFYUI_DIR / "custom_nodes" / "ComfyUI-AnimateDiff-Evolved" / "models"),
))

# --- Генерація фото: максимум якості в межах SD1.5 ---
# Апскейл після генерації (справжня ESRGAN-модель, а не просто resize) —
# пряме підвищення width/height у SD1.5 понад ~576px дає артефакти
# "дві голови", тому роздільність тягнемо апскейлом, а не латентом.
GEN_WIDTH = int(os.environ.get("COMPF_GEN_WIDTH", "512"))
GEN_HEIGHT = int(os.environ.get("COMPF_GEN_HEIGHT", "512"))
GEN_STEPS = int(os.environ.get("COMPF_GEN_STEPS", "40"))
GEN_CFG = float(os.environ.get("COMPF_GEN_CFG", "7.0"))
GEN_SAMPLER = os.environ.get("COMPF_GEN_SAMPLER", "dpmpp_2m")
GEN_SCHEDULER = os.environ.get("COMPF_GEN_SCHEDULER", "karras")
DEFAULT_LORA_STRENGTH = float(os.environ.get("COMPF_LORA_STRENGTH", "0.8"))
DEFAULT_NEGATIVE_PROMPT = os.environ.get(
    "COMPF_NEGATIVE_PROMPT",
    "lowres, worst quality, low quality, jpeg artifacts, bad anatomy, extra limbs, "
    "mutated, blurry, watermark, text, signature, deformed, disfigured, cropped, out of frame",
)

GEN_UPSCALE_ENABLED = os.environ.get("COMPF_UPSCALE_ENABLED", "1") == "1"
UPSCALE_MODEL = os.environ.get("COMPF_UPSCALE_MODEL", "RealESRGAN_x4plus.pth")
# Модель апскейлить x4; тут фінально масштабуємо результат відносно
# GEN_WIDTH/GEN_HEIGHT (2.0 => з 512 буде фінально 1024).
GEN_FINAL_SCALE = float(os.environ.get("COMPF_FINAL_SCALE", "2.0"))

# --- Відео (AnimateDiff) ---
# Дефолти розраховані на дискретну GPU (RTX 3070 12ГБ вільно тягне ці
# розміри). На MacBook без dedicated GPU опустіть через env, напр.:
#   COMPF_VIDEO_WIDTH=384 COMPF_VIDEO_HEIGHT=384 COMPF_VIDEO_FRAMES=16 COMPF_VIDEO_STEPS=16
VIDEO_WIDTH = int(os.environ.get("COMPF_VIDEO_WIDTH", "512"))
VIDEO_HEIGHT = int(os.environ.get("COMPF_VIDEO_HEIGHT", "512"))
VIDEO_FRAMES = int(os.environ.get("COMPF_VIDEO_FRAMES", "24"))
VIDEO_FPS = int(os.environ.get("COMPF_VIDEO_FPS", "8"))
VIDEO_STEPS = int(os.environ.get("COMPF_VIDEO_STEPS", "25"))
VIDEO_CFG = float(os.environ.get("COMPF_VIDEO_CFG", "6.5"))

# --- Тренування LoRA ---
# Вище rank і більше кроків = точніша схожість персонажа, ціна — довше
# тренування. На CUDA (3070) суттєво швидше, ніж на CPU/MPS у пісочниці.
TRAIN_RESOLUTION = int(os.environ.get("COMPF_TRAIN_RES", "512"))
TRAIN_STEPS = int(os.environ.get("COMPF_TRAIN_STEPS", "1500"))
TRAIN_LR = float(os.environ.get("COMPF_TRAIN_LR", "1e-4"))
TRAIN_RANK = int(os.environ.get("COMPF_TRAIN_RANK", "16"))
TRAIN_BATCH_SIZE = int(os.environ.get("COMPF_TRAIN_BATCH", "1"))
PREVIEW_POSES_AFTER_TRAINING = ["портрет анфас", "сидить", "біжить по траві"]
