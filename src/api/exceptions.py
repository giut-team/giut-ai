import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException

logger = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    detail: str


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(detail=str(exc.detail)).model_dump(),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        error = exc.errors()[0]
        location = ".".join(str(part) for part in error["loc"])
        reason = f"{location}: {error['msg']}" if location else error["msg"]
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(detail=reason).model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("요청 처리 중 예기치 않은 오류: %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(detail="서버 내부 오류가 발생했습니다.").model_dump(),
        )
