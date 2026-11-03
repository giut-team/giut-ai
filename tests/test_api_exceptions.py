from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from src.api.exceptions import register_exception_handlers


class RequestBody(BaseModel):
    count: int


def test_common_error_response():
    app = FastAPI()
    register_exception_handlers(app)

    @app.post("/items")
    def create_item(body: RequestBody):
        return body

    @app.get("/forbidden")
    def forbidden():
        raise HTTPException(status_code=403, detail="접근할 수 없습니다.")

    @app.get("/broken")
    def broken():
        raise RuntimeError("internal secret")

    with TestClient(app, raise_server_exceptions=False) as client:
        validation = client.post("/items", json={"count": "invalid"})
        http_error = client.get("/forbidden")
        not_found = client.get("/missing")
        unexpected = client.get("/broken")

    assert validation.status_code == 422
    assert "body.count" in validation.json()["detail"]
    assert set(validation.json()) == {"detail"}
    assert http_error.status_code == 403
    assert http_error.json() == {"detail": "접근할 수 없습니다."}
    assert not_found.status_code == 404
    assert set(not_found.json()) == {"detail"}
    assert unexpected.status_code == 500
    assert unexpected.json() == {"detail": "서버 내부 오류가 발생했습니다."}
