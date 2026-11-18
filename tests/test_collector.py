import asyncio
import json
import logging
import threading
from datetime import datetime
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlparse

import pytest
import requests

from src.config import settings
from src.services import collector
from src.services.extraction import InvalidContestError
from src import collector_job
from src import backend_client


def test_collect_and_post_sends_all_results_in_one_request(monkeypatch, tmp_path):
    # 부적합 활동은 수집 단계에서 InvalidContestError로 이미 제외됨
    uos_results = [{"title": "first"}, {"title": "second"}]
    contest_korea_results = [{"title": "third"}]
    sent = []
    started = []
    both_started = asyncio.Event()
    uos_args = []
    contest_korea_args = []
    run_at = datetime(2026, 10, 28, 12, 0, tzinfo=settings.TIMEZONE)
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"contest_korea_last_str_no": {"contest": 202610280003}})
    )
    monkeypatch.setattr(settings, "COLLECTOR_STATE_PATH", str(state_path))

    async def fake_collect_uos(start, end):
        uos_args.append((start, end))
        started.append("uos")
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        return uos_results

    async def fake_collect_contest_korea(last_str_nos):
        contest_korea_args.append(last_str_nos)
        started.append("contest_korea")
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        return contest_korea_results, {"contest": 202610280007, "activity": 1}

    def fake_post_competition(payload, x_ai_api_key):
        sent.append((payload, x_ai_api_key))

    monkeypatch.setattr(collector_job, "collect_uos", fake_collect_uos)
    monkeypatch.setattr(
        collector_job, "collect_contest_korea", fake_collect_contest_korea
    )
    monkeypatch.setattr(collector_job, "post_competition", fake_post_competition)

    asyncio.run(
        asyncio.wait_for(collector_job.collect_and_post("test-key", run_at), 3)
    )

    assert sent == [(uos_results + contest_korea_results, "test-key")]
    assert uos_args == [
        (datetime(2026, 10, 28, 0, 0, tzinfo=settings.TIMEZONE), run_at)
    ]
    # 저장값이 없는 목록은 수집 구간 시작일 0번부터
    assert contest_korea_args == [
        {"contest": 202610280003, "activity": 202610280000}
    ]
    assert json.loads(state_path.read_text()) == {
        "contest_korea_last_str_no": {"contest": 202610280007, "activity": 1}
    }


def test_collect_and_post_keeps_state_when_post_fails(monkeypatch, tmp_path):
    run_at = datetime(2026, 10, 28, 0, 0, tzinfo=settings.TIMEZONE)
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(settings, "COLLECTOR_STATE_PATH", str(state_path))

    async def fake_collect_uos(start, end):
        return []

    async def fake_collect_contest_korea(last_str_nos):
        return [{"title": "x"}], {"contest": 202610280009, "activity": 202610280009}

    def fake_post_competition(payload, x_ai_api_key):
        raise RuntimeError("backend down")

    monkeypatch.setattr(collector_job, "collect_uos", fake_collect_uos)
    monkeypatch.setattr(
        collector_job, "collect_contest_korea", fake_collect_contest_korea
    )
    monkeypatch.setattr(collector_job, "post_competition", fake_post_competition)

    with pytest.raises(RuntimeError):
        asyncio.run(collector_job.collect_and_post("test-key", run_at))

    assert not state_path.exists()


def test_collect_and_post_skips_request_when_nothing_collected(monkeypatch, tmp_path):
    run_at = datetime(2026, 10, 28, 0, 0, tzinfo=settings.TIMEZONE)
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(settings, "COLLECTOR_STATE_PATH", str(state_path))
    sent = []

    async def fake_collect_uos(start, end):
        return []

    async def fake_collect_contest_korea(last_str_nos):
        # 새 게시물은 있었지만 모두 필터링된 경우
        return [], {"contest": 202610280009, "activity": 202610280000}

    monkeypatch.setattr(collector_job, "collect_uos", fake_collect_uos)
    monkeypatch.setattr(
        collector_job, "collect_contest_korea", fake_collect_contest_korea
    )
    monkeypatch.setattr(
        collector_job, "post_competition", lambda *args: sent.append(args)
    )

    asyncio.run(collector_job.collect_and_post("test-key", run_at))

    assert sent == []
    # 필터링된 게시물을 다음 회차에 다시 확인하지 않도록 번호는 갱신
    assert json.loads(state_path.read_text()) == {
        "contest_korea_last_str_no": {
            "contest": 202610280009,
            "activity": 202610280000,
        }
    }


UOS_WINDOW = (
    datetime(2026, 10, 28, 0, 0, tzinfo=settings.TIMEZONE),
    datetime(2026, 10, 28, 12, 0, tzinfo=settings.TIMEZONE),
)


def test_collect_uos_fetches_feeds_and_extracts_pages_concurrently(monkeypatch):
    feed_barrier = threading.Barrier(2)
    extraction_started = []
    both_started = asyncio.Event()

    def fake_parse_uos_rss(url):
        feed_barrier.wait(timeout=3)
        title = "general" if url == collector.UOS_RSS_GENERAL else "haksa"
        published_date = (
            "Wed, 28 Oct 2026 09:00:00 +0900"
            if title == "general"
            else "Tue, 27 Oct 2026 15:00:00 +0000"  # 10/28 00:00 KST (구간 시작 포함)
        )
        excluded_date = (
            "Wed, 28 Oct 2026 12:00:00 +0900"  # 구간 끝 미포함
            if title == "general"
            else "Tue, 27 Oct 2026 23:59:59 +0900"
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

    async def fake_extract_uos_page(page):
        extraction_started.append(page["title"])
        if len(extraction_started) == 2:
            both_started.set()
        await both_started.wait()
        return {"title": page["title"], "is_eligible": True}

    monkeypatch.setattr(collector, "parse_uos_rss", fake_parse_uos_rss)
    monkeypatch.setattr(collector, "extract_uos_page", fake_extract_uos_page)

    results = asyncio.run(asyncio.wait_for(collector.collect_uos(*UOS_WINDOW), 5))

    assert [result["title"] for result in results] == ["general", "haksa"]
    assert extraction_started == ["general", "haksa"]


def test_collect_uos_skips_pages_that_fail_extraction(monkeypatch):
    def fake_parse_uos_rss(url):
        title = "general" if url == collector.UOS_RSS_GENERAL else "haksa"
        return [
            {
                "title": title,
                "body": "body",
                "images": [],
                "published_date": "Wed, 28 Oct 2026 09:00:00 +0900",
                "source_url": f"https://www.uos.ac.kr/{title}",
            }
        ]

    async def fake_extract_uos_page(page):
        if page["title"] == "general":
            raise RuntimeError("extraction failed")
        return {"title": page["title"]}

    monkeypatch.setattr(collector, "parse_uos_rss", fake_parse_uos_rss)
    monkeypatch.setattr(collector, "extract_uos_page", fake_extract_uos_page)

    results = asyncio.run(asyncio.wait_for(collector.collect_uos(*UOS_WINDOW), 5))

    assert results == [{"title": "haksa"}]


def test_collect_uos_logs_filtered_pages_as_info_without_traceback(
    monkeypatch, caplog
):
    def fake_parse_uos_rss(url):
        title = "general" if url == collector.UOS_RSS_GENERAL else "haksa"
        return [
            {
                "title": title,
                "published_date": "Wed, 28 Oct 2026 09:00:00 +0900",
                "source_url": f"https://www.uos.ac.kr/{title}",
            }
        ]

    async def fake_extract_uos_page(page):
        if page["title"] == "general":
            raise InvalidContestError("팀으로 참가할 수 없는 활동입니다.")
        raise RuntimeError("unexpected")

    monkeypatch.setattr(collector, "parse_uos_rss", fake_parse_uos_rss)
    monkeypatch.setattr(collector, "extract_uos_page", fake_extract_uos_page)

    with caplog.at_level(logging.INFO, logger=collector.logger.name):
        results = asyncio.run(asyncio.wait_for(collector.collect_uos(*UOS_WINDOW), 5))

    assert results == []
    filtered = [r for r in caplog.records if "필터링됨" in r.getMessage()]
    failed = [r for r in caplog.records if "추출 실패" in r.getMessage()]
    assert [(r.levelno, r.exc_info) for r in filtered] == [(logging.INFO, None)]
    assert "팀으로 참가할 수 없는 활동입니다." in filtered[0].getMessage()
    assert [r.levelno for r in failed] == [logging.ERROR]
    assert failed[0].exc_info is not None


def test_collect_contest_korea_filters_numbers_after_last_across_pages(monkeypatch):
    requested = []
    parsed = []
    list_url = collector.CONTEST_KOREA_CONTEST_LIST.split("?")[0]

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
                      <div class='title'><a href='view.php?int_gbn=1&amp;str_no=202610260001'>older</a></div>
                      <div class='title'><a href='view.php?int_gbn=1&amp;str_no=202610270002'>yesterday</a></div>
                    </div>
                    <div class='pagination'><a href='?int_gbn=1&page=2'>2</a></div>
                """
            else:
                html = """
                    <div class='list_style_2'>
                      <div class='title'><a href='view.php?int_gbn=2&amp;str_no=202610260006'>older</a></div>
                      <div class='title'><a href='view.php?int_gbn=2&amp;str_no=202610270005'>yesterday</a></div>
                    </div>
                """
        else:
            html = """
                <div class='list_style_2'>
                  <div class='title'><a href='view.php?int_gbn=1&amp;str_no=202610260004'>older</a></div>
                  <div class='title'><a href='view.php?int_gbn=1&amp;str_no=202610270003'>yesterday</a></div>
                </div>
            """
        return FakeResponse(url, html)

    def fake_parse(url):
        parsed.append(url)
        return {"title": url.rsplit("=", 1)[-1]}

    async def fake_extract_page(page):
        return page

    monkeypatch.setattr(collector.requests, "get", fake_get)
    monkeypatch.setattr(collector, "parse_contest_korea", fake_parse)
    monkeypatch.setattr(collector, "extract_contest_korea_page", fake_extract_page)

    results, last_str_nos = asyncio.run(
        collector.collect_contest_korea(
            {"contest": 202610270001, "activity": 202610270005}
        )
    )

    assert len(requested) == 3
    assert {params["int_gbn"] for url, params in requested if params} == {"1", "2"}
    assert all(url == list_url for url, params in requested if params)
    # 마지막 번호와 같거나 작은 게시물은 제외
    assert [url.rsplit("=", 1)[-1] for url in parsed] == [
        "202610280001",
        "202610270002",
        "202610270003",
    ]
    assert [result["source_url"] for result in results] == parsed
    # 새 게시물이 없는 목록은 기존 값 유지
    assert last_str_nos == {"contest": 202610280001, "activity": 202610270005}


@pytest.mark.parametrize("int_gbn", ["1", "2"])
def test_contest_korea_scans_only_first_30_entries(monkeypatch, int_gbn):
    requested_pages = []
    list_url = f"https://www.contestkorea.com/sub/list.php?int_gbn={int_gbn}"
    filters = {"int_gbn": int_gbn, **collector.CONTEST_KOREA_FILTERS}

    def fake_get(url, *, params=None, headers, timeout):
        if params:
            url = f"{url}?{urlencode(params)}"
        query = parse_qs(urlparse(url).query)
        for key, value in filters.items():
            assert query[key] == [value]
        page = int(query.get("page", ["1"])[0])
        requested_pages.append(page)
        assert page <= 3
        entries = []
        for position in range((page - 1) * 12 + 1, page * 12 + 1):
            # The last allowed entry matches; the next entry must be ignored.
            prefix = "20261109" if position >= 30 else "20261108"
            number = f"{prefix}{position:04d}"
            entries.append(
                f"<div class='title'><a href='view.php?int_gbn={int_gbn}"
                f"&amp;str_no={number}'>{position}</a></div>"
            )
        next_url = "?" + urlencode({**filters, "page": page + 1})
        html = (
            "<div class='list_style_2'>" + "".join(entries) + "</div>"
            f"<div class='pagination'><a href='{next_url}'>next</a></div>"
        )
        return SimpleNamespace(
            url=url, content=html.encode(), raise_for_status=lambda: None
        )

    monkeypatch.setattr(collector.requests, "get", fake_get)
    results = asyncio.run(
        collector._contest_korea_detail_urls(list_url, 202611090000)
    )

    assert requested_pages == [1, 2, 3]
    assert results == [
        f"https://www.contestkorea.com/sub/view.php?int_gbn={int_gbn}"
        "&str_no=202611090030"
    ]


@pytest.mark.parametrize(
    ("hour", "minute", "expected_delay"),
    [(23, 59, 60), (11, 59, 60), (0, 0, 12 * 3600), (12, 0, 12 * 3600)],
)
def test_daily_collector_waits_until_next_seoul_noon_or_midnight(
    monkeypatch, hour, minute, expected_delay
):
    current_time = datetime(2026, 10, 28, hour, minute, tzinfo=settings.TIMEZONE)
    delays = []

    class FakeDateTime:
        @staticmethod
        def now(tz):
            assert tz == settings.TIMEZONE
            return current_time

        combine = staticmethod(datetime.combine)

    async def fake_sleep(seconds):
        delays.append(seconds)
        raise asyncio.CancelledError

    monkeypatch.setattr(collector_job, "datetime", FakeDateTime)
    monkeypatch.setattr(collector_job.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(collector_job.run_daily_collector("test-key"))

    assert delays == [expected_delay]


def test_daily_collector_rejects_blank_api_key():
    with pytest.raises(ValueError):
        asyncio.run(collector_job.run_daily_collector("  "))


class FakeBackendResponse:
    ok = True
    status_code = 201
    text = "{}"

    def json(self):
        return {}

    def raise_for_status(self):
        pass


def test_post_competition_sends_camel_case_list_and_logs_request(monkeypatch):
    payloads = [{"title": "first"}, {"title": "second", "is_team": True}]
    requests_sent = []
    logged = []

    def fake_post(url, *, json, headers, timeout):
        requests_sent.append((url, json, headers, timeout))
        return FakeBackendResponse()

    monkeypatch.setattr(backend_client.requests, "post", fake_post)
    monkeypatch.setattr(
        backend_client,
        "write_backend_request_log",
        lambda payload, **kwargs: logged.append((payload, kwargs)),
    )

    backend_client.post_competition(payloads, " test-key ")

    [(url, sent_json, headers, timeout)] = requests_sent
    assert url == backend_client.COMPETITIONS_API_URL
    assert [item["title"] for item in sent_json] == ["first", "second"]
    assert sent_json[1]["isTeam"] is True
    assert "hostOrganization" in sent_json[0]
    assert headers == {"X-AI-API-Key": "test-key"}
    assert timeout == 10.0

    [(logged_payload, logged_kwargs)] = logged
    assert logged_payload == sent_json
    assert logged_kwargs["url"] == backend_client.COMPETITIONS_API_URL
    assert isinstance(logged_kwargs["response"], FakeBackendResponse)
    assert logged_kwargs["error"] is None


def test_post_competition_logs_request_error_and_reraises(monkeypatch):
    logged = []

    def fake_post(url, *, json, headers, timeout):
        raise requests.ConnectionError("backend down")

    monkeypatch.setattr(backend_client.requests, "post", fake_post)
    monkeypatch.setattr(
        backend_client,
        "write_backend_request_log",
        lambda payload, **kwargs: logged.append(kwargs),
    )

    with pytest.raises(requests.ConnectionError):
        backend_client.post_competition([{"title": "first"}], "test-key")

    [logged_kwargs] = logged
    assert logged_kwargs["response"] is None
    assert isinstance(logged_kwargs["error"], requests.ConnectionError)


def test_post_competition_rejects_blank_api_key():
    with pytest.raises(ValueError):
        backend_client.post_competition([], " ")
