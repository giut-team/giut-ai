from datetime import date
from typing import Iterable, Optional
from pydantic import BaseModel, ConfigDict


class CompetitionExtractRequest(BaseModel):
    url: str


class CompetitionExtractResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    title: Optional[str]
    host_organization: Optional[str]
    target_participants: Optional[str]
    is_for_enrolled_only: Optional[bool]
    application_start_at: Optional[date]
    application_end_at: Optional[date]
    total_prize_amount: Optional[int]
    prize: Optional[str]
    activity_kind: Optional[str]
    category: list[str]
    is_team: Optional[bool]
    is_team_evidence: Optional[str]  # 팀 참가 가능하다고 판단한 근거 문장
    summary: Optional[str]
    links: Optional[Iterable[str]]
