from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
import pytest

from src.api import middleware
from src.api.exceptions import register_exception_handlers
from src.api.middleware import ApiRequestLogMiddleware
from src.utils.log_handler import _body_for_log


class RequestBody(BaseModel):
    url: str


@pytest.fixture
def client_and_logs(monkeypatch):
    logged = []
    monkeypatch.setattr(
        middleware, "write_api_request_log", lambda **kwargs: logged.append(kwargs)
    )

    app = FastAPI()
    app.add_middleware(ApiRequestLogMiddleware)
    register_exception_handlers(app)

    @app.post("/api/extract")
    def extract(body: RequestBody):
        return {"title": "공모전", "url": body.url}

    @app.get("/api/forbidden")
    def forbidden():
        raise HTTPException(status_code=403, detail="접근할 수 없습니다.")

    @app.get("/api/broken")
    def broken():
        raise RuntimeError("boom")

    @app.get("/health")
    def health():
        return {"ok": True}

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, logged


def test_logs_request_and_response_bodies(client_and_logs):
    client, logged = client_and_logs

    response = client.post("/api/extract?debug=1", json={"url": "https://example.com"})

    assert response.json() == {"title": "공모전", "url": "https://example.com"}
    [log] = logged
    assert log["method"] == "POST"
    assert log["path"] == "/api/extract?debug=1"
    assert log["status_code"] == 200
    assert log["error"] is None
    assert _body_for_log(log["request_body"]) == {"url": "https://example.com"}
    assert _body_for_log(log["response_body"]) == response.json()
    assert log["received_at"] <= log["responded_at"]
    assert log["received_at"].utcoffset().total_seconds() == 9 * 3600


def test_logs_handled_error_responses(client_and_logs):
    client, logged = client_and_logs

    validation = client.post("/api/extract", json={"url": 1})
    forbidden = client.get("/api/forbidden")

    assert [log["status_code"] for log in logged] == [422, 403]
    assert _body_for_log(logged[0]["response_body"]) == validation.json()
    assert _body_for_log(logged[1]["response_body"]) == forbidden.json()
    assert all(log["error"] is None for log in logged)


def test_logs_unhandled_error_as_500_and_still_returns_error_response(
    client_and_logs,
):
    client, logged = client_and_logs

    response = client.get("/api/broken")

    assert response.status_code == 500
    assert response.json() == {"detail": "서버 내부 오류가 발생했습니다."}
    [log] = logged
    assert log["status_code"] == 500
    assert isinstance(log["error"], RuntimeError)
    assert log["request_body"] is None


def test_skips_paths_outside_api_prefix(client_and_logs):
    client, logged = client_and_logs

    assert client.get("/health").json() == {"ok": True}
    assert logged == []
