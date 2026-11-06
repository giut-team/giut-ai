import json
import os
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts.url_list import TEST_URLS
from src.parsers.uos_html_parser import parse_uos_html
from src.parsers.uos_rss_parser import parse_uos_rss
from src.services.collector import (
    UOS_RSS_GENERAL,
    UOS_RSS_HAKSA,
    _uos_link_context_blocks,
)
from src.services.extraction import _link_context_blocks, filter_official_url


def test_filter_official_url():
    with open("data/text/test/link_block_test.json", "r") as f:
        link_blocks = (json.load(f))["links"]

        for idx, source_url in enumerate(TEST_URLS["COMPETITION"][:]):
            blocks = [block for block in link_blocks if block["page_id"] == idx]
            official_url = filter_official_url(_link_context_blocks(blocks), source_url)
            print(idx, ":", official_url)
            assert type(official_url) == str or official_url is None


def test_filter_official_url_from_uos_notices(monkeypatch):
    """RSS의 여러 공지를 시립대 전용 HTML 파서로 읽어 공식 링크 판별"""
    rss = b"""<?xml version="1.0" encoding="utf-8"?>
    <rss version="2.0"><channel><title>UOS notices</title>
      <item><title>Official page</title><link>https://www.uos.ac.kr/korNotice/view.do?seq=1</link></item>
      <item><title>Other links</title><link>https://www.uos.ac.kr/korNotice/view.do?seq=2</link></item>
      <item><title>Organizer page</title><link>https://www.uos.ac.kr/korNotice/view.do?seq=3</link></item>
    </channel></rss>"""
    bodies = {
        "1": """
            <p>공모전 홈페이지: https://contest.example.org/notice / 소개 영상: https://youtu.be/example</p>
            <p>온라인 접수: https://forms.gle/example</p>
            <p>공지 목록: https://www.uos.ac.kr/korNotice/list.do</p>
        """,
        "2": """
            <p>공지 목록: https://www.uos.ac.kr/korNotice/list.do</p>
            <p>인스타그램: https://www.instagram.com/example</p>
            <p>신청: https://forms.gle/example</p>
        """,
        "3": """
            <p>주관기관 안내</p>
            <p>https://organizer.example.org/contest</p>
            <p>공지 목록: https://www.uos.ac.kr/korNotice/list.do</p>
        """,
    }

    def fake_get(url, **kwargs):
        if url == UOS_RSS_GENERAL:
            return SimpleNamespace(content=rss, raise_for_status=lambda: None)
        seq = parse_qs(urlsplit(url).query)["seq"][0]
        html = (
            '<div class="view-bx"><div class="vw-tibx"><h3>시립대 공지</h3></div>'
            f'<div class="vw-con">{bodies[seq]}</div></div>'
        )
        return SimpleNamespace(text=html, raise_for_status=lambda: None)

    monkeypatch.setattr("src.parsers.uos_rss_parser.requests.get", fake_get)
    pages = parse_uos_rss(UOS_RSS_GENERAL)
    assert len(pages) == 3
    expected = [
        "https://contest.example.org/notice",
        None,
        "https://organizer.example.org/contest",
    ]

    for rss_page, expected_url in zip(pages, expected, strict=True):
        page = parse_uos_html(rss_page["source_url"])
        full_text = page["title"] + "\n" + page["body"]
        blocks = _uos_link_context_blocks(full_text)
        assert isinstance(blocks, list), rss_page["title"]
        result = filter_official_url(blocks, page["source_url"])
        assert result == expected_url, rss_page["title"]


@pytest.mark.skipif(
    os.environ.get("UOS_LIVE_TEST") != "1",
    reason="실제 RSS 접근은 UOS_LIVE_TEST=1에서만 실행",
)
def test_filter_official_url_from_live_uos_rss():
    """실제 시립대 RSS의 공지를 읽고 선택된 URL 출력"""
    pages = parse_uos_rss(UOS_RSS_GENERAL) + parse_uos_rss(UOS_RSS_HAKSA)
    source_urls = list(
        dict.fromkeys(page["source_url"] for page in pages if page["source_url"])
    )
    assert source_urls

    for source_url in source_urls:
        page = parse_uos_html(source_url)
        full_text = page["title"] + "\n" + page["body"]
        blocks = _uos_link_context_blocks(full_text)
        assert isinstance(blocks, list), source_url
        result = filter_official_url(blocks, source_url)
        print(f"{page['title']} | {source_url} -> {result}")
        assert result is None or result.startswith(("http://", "https://"))
