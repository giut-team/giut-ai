from datetime import date
from typing import Optional
from pydantic import BaseModel, ConfigDict


class CompetitionExtractRequest(BaseModel):
    url: str


class UrlItem(BaseModel):
    type: str
    url: str
    primary: bool


class CompetitionExtractResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    title: Optional[str] = None
    host_organization: Optional[str] = None
    target_participants: Optional[str] = None
    is_for_enrolled_only: Optional[bool] = None
    application_start_at: Optional[date] = None
    application_end_at: Optional[date] = None
    total_prize_amount: Optional[int] = None
    prize_info: Optional[str] = None
    activity_kind: Optional[str] = None
    category: Optional[list[str]] = None
    is_team: Optional[bool] = None
    summary: Optional[str] = None
    images: Optional[list[str]] = None
    urls: Optional[list[UrlItem] | str] = None
