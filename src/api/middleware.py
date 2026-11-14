from datetime import datetime

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.config import settings
from src.utils.log_handler import write_api_request_log


class ApiRequestLogMiddleware:
    """`path_prefix`로 시작하는 HTTP 요청의 요청·응답 본문을 api_requests.jsonl에 기록

    예외 핸들러가 만든 4xx 응답도 그대로 기록되며, 처리되지 않은 예외는 500으로 기록한 뒤
    다시 던져 바깥의 ServerErrorMiddleware가 응답을 만들도록 함
    """

    def __init__(self, app: ASGIApp, path_prefix: str = "/api") -> None:
        self.app = app
        self.path_prefix = path_prefix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(self.path_prefix):
            await self.app(scope, receive, send)
            return

        received_at = datetime.now(settings.TIMEZONE)
        request_chunks: list[bytes] = []
        response_chunks: list[bytes] = []
        status_code: int | None = None
        error: Exception | None = None

        async def receive_and_capture() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                request_chunks.append(message.get("body", b""))
            return message

        async def send_and_capture(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "http.response.body":
                response_chunks.append(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive_and_capture, send_and_capture)
        except Exception as exc:
            error = exc
            raise
        finally:
            path = scope["path"]
            if scope.get("query_string"):
                path += "?" + scope["query_string"].decode("latin-1")
            write_api_request_log(
                method=scope["method"],
                path=path,
                request_body=b"".join(request_chunks) or None,
                received_at=received_at,
                responded_at=datetime.now(settings.TIMEZONE),
                status_code=status_code if status_code is not None else (
                    500 if error is not None else None
                ),
                response_body=b"".join(response_chunks) or None,
                error=error,
            )
