from urllib.parse import urlparse

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
import pytest

from src.parsers.html_parser import HtmlParser


@pytest.fixture
async def rendered_parser(monkeypatch):
    parser = HtmlParser(timeout=4)
    documents = {}
    visited = []
    monkeypatch.setattr(parser, "_validate_url", lambda url: None)

    async def serve_document(route):
        path = urlparse(route.request.url).path
        visited.append(path)
        document = documents.get(path)
        if document is None:
            await route.abort()
        else:
            document = {
                **document,
                "body": document["body"].replace(
                    "<html>", '<html><head><meta charset="utf-8"></head>'
                ),
            }
            await route.fulfill(content_type="text/html; charset=utf-8", **document)

    monkeypatch.setattr(parser, "_guard_route", serve_document)
    try:
        yield parser, documents, visited
    finally:
        await parser.close()


async def test_waits_for_challenge_redirect_and_delayed_content(rendered_parser):
    parser, documents, visited = rendered_parser
    documents["/challenge"] = {
        "status": 202,
        "headers": {"x-amzn-waf-action": "challenge"},
        "body": """<html><body><div>Checking your browser</div>
            <script>setTimeout(() => location.href = '/actual', 800);</script>
            </body></html>""",
    }
    documents["/actual"] = {
        "body": """<html><body><nav>메뉴</nav><main></main>
            <script>
                setTimeout(() => {
                    document.querySelector('main').innerHTML = '<p>대회 본문 준비 완료</p>';
                }, 400);
            </script></body></html>""",
    }

    result = await parser.parse_url("https://example.com/challenge", render=True)

    assert "/actual" in visited
    assert [block["text"] for block in result.text] == ["대회 본문 준비 완료"]
    assert not parser._browser.contexts


async def test_waits_for_main_and_content_stability_ignoring_changing_navigation(
    rendered_parser,
):
    parser, documents, _ = rendered_parser
    documents["/delayed"] = {
        "body": """<html><body><nav>메뉴</nav><main></main>
            <script>
                setInterval(() => {
                    document.querySelector('nav').textContent = String(performance.now());
                }, 50);
                setTimeout(() => {
                    document.querySelector('main').innerHTML = '<p>중간 내용</p>';
                }, 200);
                setTimeout(() => {
                    document.querySelector('main').innerHTML = '<p>최종 본문</p>';
                }, 550);
            </script></body></html>""",
    }

    result = await parser.parse_url("https://example.com/delayed")

    assert [block["text"] for block in result.text] == ["최종 본문"]


async def test_short_page_does_not_require_minimum_text_length(rendered_parser):
    parser, documents, _ = rendered_parser
    documents["/short"] = {"body": "<html><body><p>짧은 공지</p></body></html>"}

    result = await parser.parse_url("https://example.com/short")

    assert [block["text"] for block in result.text] == ["짧은 공지"]


async def test_image_only_page_is_ready_and_image_dimensions_are_measured(rendered_parser):
    parser, documents, _ = rendered_parser
    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j4X8AAAAASUVORK5CYII="
    )
    documents["/image"] = {
        "body": f'<html><body><main><img src="{image}" width="20" height="30"></main></body></html>'
    }

    result = await parser.parse_url("https://example.com/image")

    assert len(result.images) == 1
    assert result.images[0]["size"]["original_width"] == 1
    assert result.images[0]["size"]["display_area"] == 600


@pytest.mark.parametrize("challenge", [False, True])
async def test_missing_content_times_out_and_closes_page(rendered_parser, challenge):
    parser, documents, _ = rendered_parser
    parser.timeout = 1
    documents["/empty"] = (
        {
            "status": 202,
            "headers": {"x-amzn-waf-action": "challenge"},
            "body": "<html><body><div>Checking your browser</div></body></html>",
        }
        if challenge
        else {
            "body": """<html><body><nav>메뉴만 표시</nav><main></main>
                <noscript>JavaScript required</noscript>
                <script>const placeholder = '아직 본문 없음';</script>
                </body></html>"""
        }
    )

    with pytest.raises(PlaywrightTimeoutError):
        await parser.parse_url("https://example.com/empty")

    assert not parser._browser.contexts
