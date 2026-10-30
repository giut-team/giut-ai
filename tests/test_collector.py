import asyncio
import threading
from datetime import datetime
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

from src.services import collector
from src import backend_client


def test_collect_and_post_uos_sends_results_in_one_request(monkeypatch):
    payloads = [{"title": "first"}, {"title": "second"}]
    sent = []
    started = []
    both_started = asyncio.Event()

    async def fake_collect_uos():
        started.append("uos")
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        return payloads

    async def fake_collect_contest_korea():
        started.append("contest_korea")
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        return []

    def fake_post_competition(payload, bearer_token):
        sent.append((payload, bearer_token))

    monkeypatch.setattr(collector, "collect_uos", fake_collect_uos)
    monkeypatch.setattr(collector, "collect_contest_korea", fake_collect_contest_korea)
    monkeypatch.setattr(collector, "post_competition", fake_post_competition)

    asyncio.run(asyncio.wait_for(collector.collect_and_post("test-token"), 3))

    assert sent == [(payloads, "test-token")]


def test_collect_uos_fetches_feeds_and_extracts_pages_concurrently(monkeypatch):
    feed_barrier = threading.Barrier(2)
    extraction_started = []
    both_started = asyncio.Event()

    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 29, 12, 0, tzinfo=tz)

    def fake_parse_uos_rss(url):
        feed_barrier.wait(timeout=3)
        title = "general" if url == collector.UOS_RSS_GENERAL else "haksa"
        published_date = (
            "Wed, 28 Oct 2026 09:00:00 +0900"
            if title == "general"
            else "Tue, 27 Oct 2026 16:30:00 +0000"
        )
        excluded_date = (
            "Thu, 29 Oct 2026 09:00:00 +0900"
            if title == "general"
            else "Tue, 27 Oct 2026 09:00:00 +0900"
        )
        return [
            {
                "title": title,
                "body": "body",
                "images": [],
                "published_date": published_date,
            },
            {
                "title": "excluded",
                "body": "body",
                "images": [],
                "published_date": excluded_date,
            },
            {
                "title": "undated",
                "body": "body",
                "images": [],
                "published_date": "",
            },
        ]

    async def fake_extract_fields_from_text(text):
        extraction_started.append(text)
        if len(extraction_started) == 2:
            both_started.set()
        await both_started.wait()
        return (
            SimpleNamespace(
                to_dict=lambda: {
                    "title": text,
                    "target_participants": "대학생만 가능",
                    "activity_kind": "공모전",
                    "is_team": True,
                }
            ),
            None,
        )

    monkeypatch.setattr(collector, "datetime", FakeDateTime)
    monkeypatch.setattr(collector, "parse_uos_rss", fake_parse_uos_rss)
    monkeypatch.setattr(
        collector, "extract_fields_from_text", fake_extract_fields_from_text
    )

    results = asyncio.run(asyncio.wait_for(collector.collect_uos(), 5))

    assert [result["title"] for result in results] == [
        "general\nbody",
        "haksa\nbody",
    ]
    assert all(result["is_team"] for result in results)
    assert extraction_started == ["general\nbody", "haksa\nbody"]


def test_collect_contest_korea_filters_yesterday_across_pages(monkeypatch):
    requested = []
    parsed = []
    list_url = collector.CONTEST_KOREA_CONTEST_LIST.split("?")[0]

    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 28, 0, 0, tzinfo=tz)

    class FakeResponse:
        def __init__(self, url, html):
            self.url = url
            self.content = html.encode()

        def raise_for_status(self):
            pass

    def fake_get(url, *, params=None, headers, timeout):
        requested.append((url, params))
        if params:
            assert {k: v for k, v in params.items() if k != "int_gbn"} == (
                collector.CONTEST_KOREA_FILTERS
            )
            url = f"{url}?{urlencode(params)}"
            if params["int_gbn"] == "1":
                html = """
                    <div class='list_style_2'>
                      <div class='title'><a href='view.php?int_gbn=1&amp;str_no=202610280001'>today</a></div>
                      <div class='title'><a href='view.php?int_gbn=1&amp;str_no=202610270002'>yesterday</a></div>
                    </div>
                    <div class='pagination'><a href='?int_gbn=1&page=2'>2</a></div>
                """
            else:
                html = """
                    <div class='list_style_2'>
                      <div class='title'><a href='view.php?int_gbn=2&amp;str_no=202610270005'>yesterday</a></div>
                      <div class='title'><a href='view.php?int_gbn=2&amp;str_no=202610260006'>older</a></div>
                    </div>
                """
        else:
            html = """
                <div class='list_style_2'>
                  <div class='title'><a href='view.php?int_gbn=1&amp;str_no=202610270003'>yesterday</a></div>
                  <div class='title'><a href='view.php?int_gbn=1&amp;str_no=202610260004'>older</a></div>
                </div>
            """
        return FakeResponse(url, html)

    def fake_parse(url):
        parsed.append(url)
        return {"title": url.rsplit("=", 1)[-1]}

    async def fake_extract_page(page, semaphore):
        return page

    monkeypatch.setattr(collector, "datetime", FakeDateTime)
    monkeypatch.setattr(collector.requests, "get", fake_get)
    monkeypatch.setattr(collector, "parse_contest_korea", fake_parse)
    monkeypatch.setattr(collector, "_extract_contest_korea_page", fake_extract_page)

    results = asyncio.run(collector.collect_contest_korea())

    assert len(requested) == 3
    assert {params["int_gbn"] for url, params in requested if params} == {"1", "2"}
    assert all(url == list_url for url, params in requested if params)
    assert [url.rsplit("=", 1)[-1] for url in parsed] == [
        "202610270002",
        "202610270003",
        "202610270005",
    ]
    assert [result["url"] for result in results] == parsed


def test_daily_collector_waits_until_seoul_midnight(monkeypatch):
    current_time = datetime(2026, 10, 28, 23, 59, tzinfo=collector.SEOUL_TIMEZONE)
    delays = []

    class FakeDateTime:
        @staticmethod
        def now(tz):
            assert tz == collector.SEOUL_TIMEZONE
            return current_time

        combine = staticmethod(datetime.combine)

    async def fake_sleep(seconds):
        delays.append(seconds)
        raise asyncio.CancelledError

    monkeypatch.setattr(collector, "datetime", FakeDateTime)
    monkeypatch.setattr(collector.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(collector.run_daily_collector("test-token"))

    assert delays == [60]


def test_post_competition_sends_list_as_json(monkeypatch):
    payloads = [{"title": "first"}, {"title": "second"}]
    requests_sent = []

    class FakeResponse:
        def raise_for_status(self):
            pass

    def fake_post(url, *, json, headers, timeout):
        requests_sent.append((url, json, headers, timeout))
        return FakeResponse()

    monkeypatch.setattr(backend_client.requests, "post", fake_post)

    backend_client.post_competition(payloads, "test-token")

    assert requests_sent == [
        (
            backend_client.COMPETITIONS_URL,
            payloads,
            {"Authorization": "Bearer test-token"},
            10.0,
        )
    ]
