"""
ЗАГЛУШКА деплою в Instagram — за проханням користувача логіку поки не
реалізовуємо, лише закладаємо інтерфейс/модель даних, щоб було куди
під'єднати реальну публікацію пізніше.

Що знадобиться для реальної реалізації (TODO, не реалізовано):
  1. Instagram Business/Creator акаунт, привʼязаний до Facebook Page.
  2. Застосунок на Meta for Developers + long-lived access token
     (дозвіл instagram_content_publish).
  3. POST /{ig-user-id}/media  (image_url або video_url, caption)
     -> creation_id. Для відео/REELS: media_type=VIDEO/REELS і опитування
     GET /{creation_id}?fields=status_code, поки не FINISHED.
  4. POST /{ig-user-id}/media_publish (creation_id) -> реальний post id.
  5. Instagram не приймає локальні файли — файл має лежати за публічним
     URL (тимчасовий хостинг / S3 / CDN). Це окрема інфраструктурна
     задача, яку теж ще не піднімали.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from ..db import get_session
from ..models import Asset, InstagramStatus

router = APIRouter(prefix="/api/assets", tags=["instagram"])


@router.post("/{asset_id}/publish/instagram")
def publish_to_instagram(asset_id: int, session: Session = Depends(get_session)):
    asset = session.get(Asset, asset_id)
    if not asset:
        raise HTTPException(404, "Asset не знайдено")

    # --- тут була б реальна публікація через Instagram Graph API ---
    asset.instagram_status = InstagramStatus.queued_stub
    session.add(asset)
    session.commit()
    session.refresh(asset)

    return {
        "ok": True,
        "not_implemented": True,
        "message": (
            "Це заглушка: реальний виклик Instagram Graph API ще не "
            "реалізовано (навмисно, за вашим запитом). Дивіться TODO у "
            "app/routers/instagram.py."
        ),
        "asset_id": asset.id,
        "instagram_status": asset.instagram_status,
    }


@router.get("/{asset_id}")
def get_asset(asset_id: int, session: Session = Depends(get_session)):
    asset = session.get(Asset, asset_id)
    if not asset:
        raise HTTPException(404, "Asset не знайдено")
    return asset
