import logging
from typing import Any, Sequence

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from starlette.exceptions import HTTPException

from src.parsers.html_parser import InaccessibleURLError

logger = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    detail: str


def _first_error_reason(errors: Sequence[Any]) -> str:
    """pydantic 검증 오류 목록 중 첫 번째를 `위치: 메시지` 형태로 변환"""
    error = errors[0]
    location = ".".join(str(part) for part in error["loc"])
    return f"{location}: {error['msg']}" if location else error["msg"]


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def handle_http_exception(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(detail=str(exc.detail)).model_dump(),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(detail=_first_error_reason(exc.errors())).model_dump(),
        )

    @app.exception_handler(InaccessibleURLError)
    async def handle_inaccessible_url(
        request: Request, exc: InaccessibleURLError
    ) -> JSONResponse:
        # 사용자에게는 원인을 노출하지 않고, 원인 예외는 서버 로그로만 남김
        logger.warning("접속할 수 없는 URL 요청: %s", exc)
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(detail="잘못된 형식의 url입니다.").model_dump(),
        )

    @app.exception_handler(ValueError)
    async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
        # pydantic ValidationError도 ValueError의 하위 클래스라 이 핸들러로 들어옴
        if isinstance(exc, ValidationError):
            reason = _first_error_reason(exc.errors())
        else:
            reason = str(exc) or "잘못된 값입니다."
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(detail=reason).model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "요청 처리 중 예기치 않은 오류: %s %s", request.method, request.url.path
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(detail="서버 내부 오류가 발생했습니다.").model_dump(),
        )
