from pydantic import BaseModel


class CompetitionExtractRequest(BaseModel):
    url: str


# TODO: 응답 DTO 형식 정의
