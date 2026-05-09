from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.auth import require_token
from app.db.models import Category
from app.db.session import get_db
from app.schemas.jobs import StyleProfileItem
from app.services.style_profiles import list_active_profiles

router = APIRouter(
    prefix="/api/style-profiles",
    tags=["style-profiles"],
    dependencies=[Depends(require_token)],
)


@router.get("", response_model=list[StyleProfileItem])
def list_style_profiles(
    category: Category | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[StyleProfileItem]:
    """Return active style profiles, optionally filtered by category.

    The UI uses this to populate the profile dropdown at upload time and the
    edit-profile dialog while a job is in AWAITING_CONFIRMATION.
    """
    return [StyleProfileItem.model_validate(p) for p in list_active_profiles(db, category)]
