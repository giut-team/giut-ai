from datetime import datetime

from dataclasses import fields
import logging
from pydantic import field_validator
from pydantic.dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ActivityKindEnum(Enum):
    CONTEST = "공모전"
    COMPETITION = "경진대회"
    HACKATHON = "해커톤"
    SUPPORTERS = "서포터즈"
    VOLUNTEER = "봉사"
    STARTUP = "창업"
    EDUCATION = "교육/프로그램"
    OVERSEAS_EXPERIENCE = "해외탐방"
    NETWORKING = "행사/네트워킹"
    ETC = "기타"
    # 여기부턴 들어갈지 말지 모름 확정안됨
    INTERNSHIP = "인턴/직무체험"


class CategoryEnum(Enum):
    IDEA = "기획/아이디어"
    MARKETING = "홍보/마케팅"
    AI_DATA = "AI/데이터"
    IT_DEVELOP = "IT/개발"
    SCIENCE_ENGINEERING = "과학/공학"
    DESIGN = "디자인"
    PHOTO_CONTENTS = "사진/영상/콘텐츠"
    LITERATURE = "문학/글쓰기"
    RESEARCH = "학술/연구"
    CULTURE_ARTS = "문화/예체능"
    ETC = "기타"


@dataclass
class ExtractedFields:
    title: Optional[str]
    host_organization: Optional[str]
    target_participants: Optional[str]
    application_start_at: Optional[datetime]
    application_end_at: Optional[datetime]
    prize: Optional[str]
    activity_kind: Optional[ActivityKindEnum]
    category: Optional[list[CategoryEnum]]
    is_team: Optional[bool]
    is_team_evidence: Optional[str]  # 팀 참가 가능하다고 판단한 근거 문장
    summary: Optional[str]

    @field_validator("activity_kind", mode="before")
    @classmethod
    def _validate_activity_kind(cls, value: object) -> Optional[ActivityKindEnum]:
        if value is None:
            return None
        try:
            return ActivityKindEnum(value)
        except (ValueError, TypeError):
            logger.warning("LLM 응답 활동종류 값 오류: %r", value)
            return None

    @field_validator("category", mode="before")
    @classmethod
    def _validate_category(cls, value: object) -> object:
        if not isinstance(value, list):
            return value

        categories = []
        for item in value:
            try:
                categories.append(CategoryEnum(item))
            except (ValueError, TypeError):
                logger.warning("LLM 응답 활동분야 값 오류: %r", item)
        return categories or None

    def format(self) -> str:
        formatted = ""
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                formatted += f"{field.name}: {value}\n"
        return formatted

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractedFields":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
