import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient
from playwright.async_api import Error as PlaywrightError
import pytest
import requests

from src.api import router as router_module
from src.api.exceptions import register_exception_handlers
from src.parsers.html_parser import HtmlParser, InaccessibleURLError

INVALID_URL_RESPONSE = {"detail": "잘못된 형식의 url입니다."}


# -- HtmlParser: 접속 단계 실패만 InaccessibleURLError로 변환 ----------------------


@pytest.mark.parametrize(
    "url",
    ["ftp://example.com/file", "not-a-url", "http://localhost/admin"],
)
@pytest.mark.parametrize("render", [True, False])
def test_invalid_or_blocked_url_raises_inaccessible(url, render):
    parser = HtmlParser()

    with pytest.raises(InaccessibleURLError):
        asyncio.run(parser.fetch(url, render=render))

    # 검증 단계에서 막혀 브라우저를 띄우지 않아야 함
    assert parser._browser is None


@pytest.mark.parametrize(
    "error",
    [requests.ConnectionError("refused"), requests.Timeout("slow")],
)
def test_static_fetch_connection_error_raises_inaccessible(monkeypatch, error):
    parser = HtmlParser()
    monkeypatch.setattr(parser, "_validate_url", lambda url: None)

    def fake_get(*args, **kwargs):
        raise error

    monkeypatch.setattr(parser.session, "get", fake_get)

    with pytest.raises(InaccessibleURLError) as exc_info:
        asyncio.run(parser.fetch("https://example.com", render=False))
    assert exc_info.value.__cause__ is error


def test_static_fetch_http_error_status_raises_inaccessible(monkeypatch):
    parser = HtmlParser()
    monkeypatch.setattr(parser, "_validate_url", lambda url: None)

    response = requests.Response()
    response.status_code = 404
    response.url = "https://example.com/missing"
    monkeypatch.setattr(parser.session, "get", lambda *args, **kwargs: response)

    with pytest.raises(InaccessibleURLError):
        asyncio.run(parser.fetch("https://example.com/missing", render=False))


class FakePage:
    def __init__(self, goto_error):
        self.goto_error = goto_error
        self.closed = False

    def on(self, event, handler):
        pass

    async def route(self, pattern, handler):
        pass

    async def goto(self, url, **kwargs):
        raise self.goto_error

    async def close(self):
        self.closed = True


def test_rendered_navigation_failure_raises_inaccessible_and_closes_page(monkeypatch):
    parser = HtmlParser()
    page = FakePage(PlaywrightError("net::ERR_CONNECTION_REFUSED"))
    monkeypatch.setattr(parser, "_validate_url", lambda url: None)

    class FakeBrowser:
        async def new_page(self, **kwargs):
            return page

    async def fake_ensure_browser():
        return FakeBrowser()

    monkeypatch.setattr(parser, "_ensure_browser", fake_ensure_browser)

    with pytest.raises(InaccessibleURLError):
        asyncio.run(parser.fetch("https://example.com", render=True))
    assert page.closed


def test_browser_launch_failure_is_not_treated_as_bad_url(monkeypatch):
    parser = HtmlParser()
    monkeypatch.setattr(parser, "_validate_url", lambda url: None)

    async def broken_ensure_browser():
        raise PlaywrightError("Executable doesn't exist")

    monkeypatch.setattr(parser, "_ensure_browser", broken_ensure_browser)

    with pytest.raises(PlaywrightError):
        asyncio.run(parser.fetch("https://example.com", render=True))


# -- API: 접속 불가 URL은 400, 그 외 내부 오류는 500 -------------------------------


@pytest.fixture
def client():
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router_module.router)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def post_extract(client, url):
    return client.post("/api/competitions/extract", json={"url": url})


@pytest.mark.parametrize(
    ("url", "parser_name"),
    [
        ("https://www.uos.ac.kr/notice/1", "parse_uos_html"),
        ("https://www.contestkorea.com/sub/view.php?str_no=1", "parse_contest_korea"),
    ],
)
def test_site_parser_connection_error_returns_400(monkeypatch, client, url, parser_name):
    def fake_parse(url):
        raise requests.HTTPError("404 Client Error")

    monkeypatch.setattr(router_module, parser_name, fake_parse)

    response = post_extract(client, url)

    assert response.status_code == 400
    assert response.json() == INVALID_URL_RESPONSE


def test_pipeline_inaccessible_url_returns_400(monkeypatch, client):
    async def fake_run_pipeline(url):
        raise InaccessibleURLError("cannot resolve host: 'nowhere.invalid'")

    monkeypatch.setattr(router_module, "run_pipeline", fake_run_pipeline)

    response = post_extract(client, "https://nowhere.invalid")

    assert response.status_code == 400
    assert response.json() == INVALID_URL_RESPONSE


def test_internal_error_after_fetch_stays_500(monkeypatch, client):
    async def fake_run_pipeline(url):
        # 접속 이후 단계(LLM 호출 등)의 requests 오류는 사용자 URL 문제가 아님
        raise requests.ConnectionError("openai down")

    monkeypatch.setattr(router_module, "run_pipeline", fake_run_pipeline)

    response = post_extract(client, "https://example.com")

    assert response.status_code == 500
    assert response.json() == {"detail": "서버 내부 오류가 발생했습니다."}
