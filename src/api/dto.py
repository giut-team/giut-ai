from datetime import date
from typing import Iterable, Optional
from pydantic import BaseModel, ConfigDict


class CompetitionExtractRequest(BaseModel):
    url: str


class CompetitionExtractResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    title: Optional[str] = None
    host_organization: Optional[str] = None
    target_participants: Optional[str] = None
    is_for_enrolled_only: Optional[bool] = None
    application_start_at: Optional[date] = None
    application_end_at: Optional[date] = None
    total_prize_amount: Optional[int] = None
    prize: Optional[str] = None
    activity_kind: Optional[str] = None
    category: Optional[list[str]] = None
    is_team: Optional[bool] = None
    summary: Optional[str] = None
    images: Optional[list[str]] = None
    source_url: str
    links: Optional[Iterable[str]] = None
