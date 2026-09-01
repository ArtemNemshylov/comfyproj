# compfui-admin — зручна адмінка над ComfyUI (POC)

Адмінка для роботи з конкретними персонажами-собаками поверх
справжнього [ComfyUI](https://github.com/comfyanonymous/ComfyUI):
генерація фото в різних позах, генерація короткого відео, тренування
"портрета" персонажа (LoRA) з памʼяттю між версіями (додав шрам —
він лишається на всіх наступних фото), і заглушка під майбутній
деплой в Instagram.

Два способи запуску — код той самий, різниться лише деплой:

- **Docker на ПК з NVIDIA GPU** (рекомендовано, зокрема перевірено під
  RTX 3070 12ГБ / 32ГБ RAM) — `docker compose up --build`, і все саме
  довантажить моделі й підніметься. Швидше й надійніше за нативний
  macOS-шлях (справжня CUDA, ніякого MPS/пісочниці).
- **Нативно на macOS** (M-серія) — окремі скрипти, детальніше нижче.

Якість під капотом навмисно піднята на максимум **у межах SD1.5**
(більше кроків, dpmpp_2m+karras, реальний ESRGAN-апскейл, LoRA rank=16 /
1500 кроків тренування) — все ще не SDXL-рівень фотореалізму, але
значно краще за початковий POC. Усі числа — в `admin/app/config.py`,
перевизначаються env-змінними `COMPF_*`.

## Архітектура

```
compfproject/
├── comfyui/                 ← справжній ComfyUI (нативний запуск; клонується setup-скриптом, у git не комітиться)
├── admin/                   ← адмінка (FastAPI + vanilla JS)
│   ├── app/                   бекенд: роутери, БД, клієнт ComfyUI, побудова workflow-графів
│   ├── training/               окремий скрипт тренування LoRA (diffusers + peft)
│   ├── static/                 фронтенд (index.html/app.js/style.css)
│   └── data/                   характери/фото/відео/LoRA/sqlite (у git не комітиться)
├── docker/                  ← Dockerfile'и + entrypoint для Docker-запуску
├── docker-compose.yml       ← оркестрація comfyui + admin контейнерів
├── docker-data/             ← томи Docker-запуску (моделі, дані) — у git не комітиться
└── scripts/                 setup_comfyui.sh, setup_admin.sh, run_all.sh, stop_all.sh (нативний macOS-шлях)
```

Компонент | Роль
---|---
**ComfyUI** | реальний рушій генерації (текст→фото, LoRA, AnimateDiff-відео). Адмінка не малює нічого сама — вона будує ComfyUI workflow-графи (JSON) і відправляє їх через HTTP API ComfyUI (`/prompt`, `/history`, `/view`).
**admin (FastAPI)** | реєстр персонажів/версій/файлів (SQLite), логіка "памʼяті" й підтвердження, запуск тренування LoRA як підпроцесу, роздача фронтенду.
**train_lora.py** | окремий Python-скрипт (diffusers + peft), тренує LoRA на завантажених фото собаки. Запускається бекендом як підпроцес (в Docker — усередині admin-контейнера), пише прогрес у JSON-файл. Автоматично обирає CUDA → MPS → CPU.

## Модель памʼяті персонажа

- **Character** — собака (імʼя, базовий опис, унікальний `trigger_token`,
  напр. `sks_rex`).
- **CharacterVersion** — "підтверджений портрет" на момент часу.
  Кожна нова версія посилається на попередню й **накопичує**:
  - усі референсні фото попередніх версій + нові;
  - весь текстовий опис (`prompt_prefix`) попередніх версій + нову
    trait-нотатку.

  Приклад: у персонажа Рекса підтверджена v1 ("рудий корги"). Потім
  зʼявився шрам — ви завантажуєте нові фото й пишете нотатку "зʼявився
  шрам на лівому вусі". Створюється v2: `prompt_prefix` = "рудий корги,
  зʼявився шрам на лівому вусі", а фото для тренування = усі фото v1 +
  нові фото зі шрамом. Після тренування й підтвердження v2 стає
  `character.current_version_id` — **усі** наступні фото й відео
  автоматично генеруються з новою LoRA і з цим текстом, тобто шрам
  "памʼятається" назавжди, поки ви не відкотитесь на іншу версію
  вручну.
- **Job** — фонове тренування (окремий процес), прогрес читається з
  JSON-файлу в `admin/data/jobs/`.
- **Asset** — конкретне згенероване фото/відео, привʼязане до версії,
  з полем `instagram_status` для заглушки деплою.

## Запуск через Docker (рекомендовано, ПК з NVIDIA GPU)

Перевірено з розрахунку на RTX 3070 12ГБ / 32ГБ RAM. Потрібно на хості:

- Docker + Docker Compose v2 (`docker compose`, не старий `docker-compose`).
- NVIDIA-драйвер + GPU-підтримка в Docker: на Linux — пакет
  [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html);
  на Windows — Docker Desktop з увімкненим WSL2-backend і GPU-підтримкою.
- Перевірка, що GPU взагалі видно Docker'у:
  ```
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```
  Якщо це не показує вашу 3070 — спершу полагодьте це, адмінка тут
  безсила (це питання драйверів/Docker, не коду).

Запуск:

```bash
git clone <ваш репозиторій> compfproject   # або git pull, якщо вже клоновано
cd compfproject
docker compose up --build
```

Перший запуск довантажує моделі (~6 ГБ: SD1.5 checkpoint, AnimateDiff
motion-модуль, ESRGAN upscaler) прямо в `docker-data/` — прогрес видно
в консолі або через `docker compose logs -f comfyui`. Наступні запуски
(`docker compose up`) моделі вже не чіпають.

Відкрийте **http://localhost:8000**.

Зупинити: `docker compose down` (дані в `docker-data/` лишаються).
Оновити код (після `git pull`): `docker compose up --build`.

### Як це влаштовано під капотом (Docker)

Два контейнери, спільна мережа compose:
- `comfyui` — сам ComfyUI, слухає `0.0.0.0:8188`, том
  `docker-data/comfyui-models` → `/app/comfyui/models` (чекпоінти,
  upscaler), мотор AnimateDiff — усередині образу.
- `admin` — ця адмінка, слухає `0.0.0.0:8000`, звертається до ComfyUI
  за іменем сервіса `http://comfyui:8188` (внутрішній DNS compose),
  тренує LoRA просто в собі (том `docker-data/admin-data` → `/data`,
  і read-only доступ до `docker-data/comfyui-models/checkpoints` для
  завантаження базової моделі під час тренування).

Усе налаштовується через `COMPF_*` env-змінні в `docker-compose.yml` —
код той самий, що й для нативного macOS-запуску (див.
`admin/app/config.py`).

## Альтернатива: нативний запуск на macOS (без Docker)

Для MacBook (M-серія) без dedicated GPU — використовує Metal/MPS
замість CUDA, окремі venv-и, довше й повільніше за Docker-шлях. Скрипти
з `scripts/` мають виконуватись у **звичайному терміналі macOS**
(Terminal.app/iTerm), не в пісочниці Cowork — вона не бачить Metal і
має обмежений доступ до huggingface.co.

```bash
cd /Users/Artem/PycharmProjects/compfproject

# 1. Клонує ComfyUI + кастомні ноди + завантажує моделі (~6 ГБ)
bash scripts/setup_comfyui.sh

# 2. Створює venv для адмінки і ставить FastAPI/diffusers/peft/torch
bash scripts/setup_admin.sh

# 3. Піднімає ComfyUI (:8188) і адмінку (:8000) у фоні
bash scripts/run_all.sh
```

Відкрийте **http://127.0.0.1:8000**. Зупинити: `bash scripts/stop_all.sh`.
Логи — `logs/comfyui.log`, `logs/admin.log`.

На MacBook варто опустити параметри якості під MPS (за замовчуванням
тепер тюнінг під дискретну GPU), напр.:
```bash
export COMPF_VIDEO_WIDTH=384 COMPF_VIDEO_HEIGHT=384 COMPF_VIDEO_FRAMES=16 COMPF_VIDEO_STEPS=16
```
перед `bash scripts/run_all.sh` (або пропишіть у `scripts/run_all.sh`).

### Якщо завантаження моделей не пройшло (обидва шляхи)

Скрипти тягнуть моделі з Hugging Face/GitHub прямими посиланнями (без
токена, репозиторії негейтовані). Якщо URL змінився/недоступний —
покладіть файл вручну:

- SD1.5: `.../models/checkpoints/v1-5-pruned-emaonly.safetensors`
- AnimateDiff motion-модуль: `.../custom_nodes/ComfyUI-AnimateDiff-Evolved/models/mm_sd_v15_v2.ckpt`
- ESRGAN upscaler: `.../models/upscale_models/RealESRGAN_x4plus.pth`

(в Docker — усередині `docker-data/comfyui-models/...`, у macOS — усередині `comfyui/models/...`)

## Як користуватись

1. **+ Новий персонаж** — завантажте першу фотографію (і, бажано,
   ще кілька) вашої собаки, вкажіть імʼя й короткий опис (порода,
   забарвлення).
2. Натисніть **"Тренувати модель (LoRA)"** — запуститься фонове
   тренування (на RTX 3070 — хвилини; на MacBook CPU/MPS — довше).
3. Коли статус стане **ready_for_review**, адмінка сама згенерує
   кілька прев'ю-поз. Перегляньте їх і натисніть **"Підтвердити
   портрет"** (або "Відхилити" і перетренуйте — кнопка тренування
   зʼявляється знову й для відхилених версій).
4. У вкладці **"Генерація фото"** пишіть будь-яку позу/сцену — модель
   генеруватиме саме цю собаку.
5. У вкладці **"Відео"** опишіть рух — генерується короткий кліп
   (AnimateDiff, той самий персонаж).
6. У вкладці **"Нова риса / перетренувати"** додайте нову деталь
   (напр. шрам) з новими фото — створиться нова версія, яку так само
   треба натренувати й підтвердити; після цього деталь буде на всіх
   майбутніх фото/відео.
7. На кожній картці фото/відео є кнопка **"Deploy → Instagram"** —
   поки що це **заглушка** (див. нижче).

## Instagram — навмисна заглушка

За вашим запитом реальну публікацію в Instagram **не реалізовано** —
лише закладено інтерфейс: `POST /api/assets/{id}/publish/instagram`
позначає asset статусом `queued_stub` і повертає `not_implemented: true`.

Що знадобиться для реальної реалізації (описано й у коді,
`admin/app/routers/instagram.py`):

1. Instagram Business/Creator акаунт, привʼязаний до Facebook Page.
2. Застосунок на Meta for Developers + long-lived access token
   (дозвіл `instagram_content_publish`).
3. `POST /{ig-user-id}/media` (image_url/video_url, caption) →
   `creation_id`. Для відео/Reels — очікування `status_code=FINISHED`.
4. `POST /{ig-user-id}/media_publish` (creation_id) → реальний post id.
5. Instagram не приймає локальні файли — потрібен публічний URL
   (тимчасовий хостинг/S3/CDN) — окрема інфраструктурна задача.

## Відомі обмеження / куди дивитись, якщо щось не працює

- **Docker: "could not select device driver" / GPU не бачиться** —
  перевірте `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
  окремо від нашого compose. Якщо це не працює — проблема в
  nvidia-container-toolkit/Docker Desktop GPU-налаштуваннях, не в коді
  адмінки.
- **AnimateDiff node_errors**: назви класів кастомних нод (`ADE_*`,
  `VHS_VideoCombine`) періодично міняються між версіями репозиторіїв.
  Якщо ComfyUI відхиляє відео-workflow — відкрийте
  `http://localhost:8188/object_info` (або
  `http://127.0.0.1:8188/object_info` для нативного шляху), знайдіть
  реальні назви й підправте словник `NODE_TYPES` на початку
  `admin/app/workflows.py` — це єдине місце, де це треба зробити.
- **Формат LoRA**: `train_lora.py` конвертує ваги в kohya-сумісний
  формат (`convert_state_dict_to_kohya` з diffusers ≥0.27) саме тому,
  що цього очікує нода `LoraLoader` у ComfyUI. Якщо ComfyUI не бачить
  LoRA або лається на ключі — перевірте версію diffusers і вміст
  `.safetensors` (`safetensors.safe_open`).
- **Якість**: дефолти (`admin/app/config.py`) тепер тюнінг "максимум у
  межах SD1.5" під дискретну GPU — 40 кроків, dpmpp_2m+karras, реальний
  ESRGAN-апскейл (512→1024), LoRA rank=16/1500 кроків тренування. Для
  ще кращої якості (SDXL) знадобиться окрема, більша переробка
  (SDXL LoRA-тренування інакше влаштоване, AnimateDiff-SDXL менш
  зрілий) — поки не робили.
- **Відео найважче**: навіть на 3070 512×512/24 кадри/25 кроків може
  рахуватись хвилину-другу; на MacBook без GPU — значно довше, там
  варто опустити параметри (див. вище).
- **Однокористувацький POC**: без авторизації, CORS відкритий на все —
  не виставляйте цей сервіс назовні як є.
