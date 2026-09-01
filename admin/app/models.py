"""
Моделі БД (SQLModel/SQLite) для compfui-admin.

Ідея "памʼяті" персонажа:
- Character — собака як сутність верхнього рівня.
- CharacterVersion — "підтверджений портрет" на певний момент часу.
  Кожна нова версія посилається на попередню (parent_version_id) і
  НАКОПИЧУЄ trait-нотатки та референсні фото попередніх версій. Тому
  якщо у v2 зʼявився шрам, то v3, v4... успадковують його автоматично —
  і через LoRA (натреновану на кумулятивних фото), і текстово (через
  prompt_prefix, який теж накопичується і завжди підмішується в промпт).
- Character.current_version_id завжди вказує на останню ПІДТВЕРДЖЕНУ
  версію — саме її lora_path + prompt_prefix використовуються при
  генерації нових фото/відео. Поки версія не підтверджена (ready_for_review),
  вона на "чернетці" і не впливає на основний потік генерації.
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional, List

from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, JSON


class CharacterStatus(str, enum.Enum):
    draft = "draft"
    training = "training"
    confirmed = "confirmed"


class VersionStatus(str, enum.Enum):
    pending = "pending"
    training = "training"
    ready_for_review = "ready_for_review"
    confirmed = "confirmed"
    rejected = "rejected"
    failed = "failed"


class JobType(str, enum.Enum):
    train_lora = "train_lora"
    generate_photo = "generate_photo"
    generate_video = "generate_video"


class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class AssetType(str, enum.Enum):
    photo = "photo"
    video = "video"


class InstagramStatus(str, enum.Enum):
    none = "none"
    queued_stub = "queued_stub"
    posted = "posted"
    failed = "failed"


class Character(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    base_description: str = ""
    trigger_token: str
    base_checkpoint: str = "v1-5-pruned-emaonly.safetensors"
    status: CharacterStatus = Field(default=CharacterStatus.draft)
    current_version_id: Optional[int] = Field(default=None, foreign_key="characterversion.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CharacterVersion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    character_id: int = Field(foreign_key="character.id")
    parent_version_id: Optional[int] = Field(default=None, foreign_key="characterversion.id")
    version_number: int

    trait_note: str = ""
    prompt_prefix: str = ""
    reference_images: List[str] = Field(default_factory=list, sa_column=Column(JSON))

    status: VersionStatus = Field(default=VersionStatus.pending)
    lora_path: Optional[str] = None
    training_job_id: Optional[int] = Field(default=None, foreign_key="job.id")

    created_at: datetime = Field(default_factory=datetime.utcnow)
    confirmed_at: Optional[datetime] = None


class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    type: JobType
    status: JobStatus = Field(default=JobStatus.queued)
    character_id: Optional[int] = Field(default=None, foreign_key="character.id")
    character_version_id: Optional[int] = Field(default=None, foreign_key="characterversion.id")
    pid: Optional[int] = None
    progress: float = 0.0
    log_path: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None


class Asset(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    character_id: int = Field(foreign_key="character.id")
    character_version_id: int = Field(foreign_key="characterversion.id")
    type: AssetType
    file_path: str
    prompt: str = ""
    negative_prompt: str = ""
    seed: Optional[int] = None
    meta: dict = Field(default_factory=dict, sa_column=Column(JSON))

    instagram_status: InstagramStatus = Field(default=InstagramStatus.none)
    instagram_post_id: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
