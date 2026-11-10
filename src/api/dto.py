from datetime import date
from typing import Annotated, Optional
from pydantic import BaseModel, ConfigDict, PlainSerializer
from pydantic.alias_generators import to_camel

from src.llm.dto import TargetParticipantsEnum, ActivityKindEnum, CategoryEnum

EnumName = PlainSerializer(lambda e: e.name, return_type=str)


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,  # snake case -> camel case 변환 함수
        validate_by_name=True,  # snake_case 키로도 입력 허용
        validate_by_alias=True,  # camelCase 키로도 입력 허용
        serialize_by_alias=True,  # 출력할 때 camelCase로
    )


class CompetitionExtractRequest(BaseModel):
    url: str


class UrlItem(CamelModel):
    type: str
    url: str | None


class CompetitionExtractResponse(CamelModel):
    title: Optional[str] = None
    host_organization: Optional[str] = None
    target_participants: Optional[Annotated[TargetParticipantsEnum, EnumName]] = None
    is_for_enrolled_only: Optional[bool] = None
    application_start_at: Optional[date] = None
    application_end_at: Optional[date] = None
    total_prize_amount: Optional[int] = None
    prize_info: Optional[str] = None
    activity_kind: Optional[Annotated[ActivityKindEnum, EnumName]] = None
    category: Optional[list[Annotated[CategoryEnum, EnumName]]] = None
    is_team: Optional[bool] = None
    summary: Optional[str] = None
    images: Optional[list[str]] = None
    urls: Optional[list[UrlItem]] = None
