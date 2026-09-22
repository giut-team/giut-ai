from datetime import datetime

from dataclasses import dataclass, fields
from enum import Enum
from typing import Optional


class CategoryEnum(Enum):
    AI_DATA = "AI/데이터"
    DEVELOP = "개발"
    IDEA = "기획/아이디어"
    DESIGN = "디자인"
    ETC = "기타"


@dataclass
class ExtractedFields:
    title: Optional[str]
    host_organization: Optional[str]
    target_participants: Optional[str]
    application_start_at: Optional[datetime]
    application_end_at: Optional[datetime]
    prize: Optional[str]
    is_competition: bool
    is_team: bool
    category: CategoryEnum
    summary: str

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
