from __future__ import annotations

from datetime import date
import re
import logging

from dataclasses import fields
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
    application_start_at: Optional[date]
    application_end_at: Optional[date]
    prize: Optional[str]
    activity_kind: Optional[ActivityKindEnum]
    category: list[CategoryEnum]
    is_team: Optional[bool]
    is_team_evidence: Optional[str]  # 팀 참가 가능하다고 판단한 근거 문장
    summary: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> ExtractedFields:
        # data 객체 내에 정의되지 않은 키가 있으면 무시
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def to_dict(cls, data: ExtractedFields) -> dict:
        return {field.name: str(getattr(data, field.name)) for field in fields(cls)}

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

    def _normalize(self, value: str) -> str:
        unnecessary_char_removed = re.sub(r"[^a-zA-Z0-9가-힣]", " ", value)
        spaces_removed = re.sub(r"\s+", " ", unnecessary_char_removed)
        return spaces_removed

    def _compare_field(self, value, target_value):
        if isinstance(value, list) and isinstance(target_value, list):
            # 카테고리 리스트에 대해서는 정답 리스트 안에 있는 것/추가/누락된 것을 각각 숫자로 바교
            results = {"EXACT": 0, "ADDED": 0, "OMITTED": 0}

            for v in value:
                if v in target_value:
                    # 키가 정답 리스트 안에 존재할 경우
                    results["EXACT"] += 1
                else:
                    # 정답 리스트안에 없을 경우. 즉 모델이 추가한 경우
                    results["ADDED"] += 1

            # 정답 리스트에 있는데 모델 응답에서는 누락된 경우
            diff = set(target_value) - set(value)
            results["OMITTED"] += len(diff)

            return results

        if value == target_value:
            # 정확히 일치할 경우 EXACT 판정
            return "EXACT"

        if isinstance(value, str) and isinstance(target_value, str):
            # 문자열 타입 정규화 후 일부 (4단어 이상) 겹칠 경우 CLOSE로 처리

            words = set(self._normalize(value).split())
            target_words = set(self._normalize(target_value).split())

            if len(words & target_words) >= 4:
                return "CLOSE"

        return "MISS"

    def compare(self, target: ExtractedFields) -> dict[str, object]:
        # 정답 라벨(target)과 모델의 실제 응답을 필드 하나씩 비교
        results = {}

        for field in fields(self):
            # 요약문 비교는 생략
            if field.name == "summary":
                continue

            value = getattr(self, field.name)
            target_value = getattr(target, field.name)

            results[field.name] = self._compare_field(value, target_value)

        return results
