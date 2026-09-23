# 콘테스트코리아 상세 페이지의 테이블 내용, 포스터 URL 추출

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag


def _text(node: Tag | None) -> str | None:
    if node is None:
        return None
    return " ".join(node.get_text(" ", strip=True).split()) or None


def _label(value: str) -> str:
    return re.sub(r"[\s.·ㆍ/&:]+", "", value)


def _absolute_url(value: str | None, base_url: str) -> str | None:
    value = (value or "").strip()
    if not value or value.startswith("#"):
        return None
    resolved = urljoin(base_url, value)
    return resolved if urlparse(resolved).scheme in {"http", "https"} else None


def _link(node: Tag | None, base_url: str) -> str | None:
    if node is None:
        return None
    for anchor in node.select("a"):
        # 홈페이지 버튼은 href="javascript:void(0)"이고 실제 주소는 val2에 있음
        for attribute in ("val2", "href"):
            url = _absolute_url(anchor.get(attribute), base_url)
            if url:
                return url
    text = _text(node)
    if text and re.match(r"https?://\S+$", text):
        return _absolute_url(text, base_url)
    return None


def _date(value: str) -> str | None:
    match = re.search(r"(\d{4})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})", value)
    if not match:
        return None
    try:
        return date(*(int(part) for part in match.groups())).isoformat()
    except ValueError:
        return None


def _detail(
    soup: BeautifulSoup, base_url: str, main_image_link: str
) -> tuple[str | None, list[str]]:
    detail = soup.select_one("div.view_detail_area")
    if detail is None:
        return None, []

    # 자식 태그부터 제거하여 중첩된 tip 요소 안전하게 처리
    for node in reversed(detail.find_all(True)):
        classes = node.get("class", [])
        is_tip = any("tip" in name.lower() for name in classes)
        is_related = node.name == "ul" and "related_blog" in classes
        is_related_heading = (
            node.name in {"h1", "h2", "h3", "h4", "h5", "h6"}
            and _text(node) == "관련 정보 및 후기"
        )
        is_inquiry = node.name == "a" and (
            node.get("id") == "btn_requ"
            or "온라인으로문의하기"
            in re.sub(
                r"\s+",
                "",
                " ".join(
                    [
                        _text(node) or "",
                        node.get("title", ""),
                        node.get("aria-label", ""),
                        *(img.get("alt", "") for img in node.select("img")),
                    ]
                ),
            )
        )
        if is_tip or is_related or is_related_heading or is_inquiry:
            node.decompose()

    # 하단 img_area의 src 추출
    image_links = []
    for img in detail.select(".img_area img[src]"):
        url = _absolute_url(img.get("src"), base_url)
        if url and url != main_image_link and url not in image_links:
            # main_image_link와 같은 사진일 경우 detail_image_links에 넣지 않음
            image_links.append(url)

    text_area = detail.select_one("div.txt")
    if text_area is None:
        return None, image_links
    for node in text_area.select("script, style, noscript"):
        node.decompose()
    lines = (
        " ".join(line.split())
        for line in text_area.get_text(separator="\n").splitlines()
    )
    description = "\n".join(line for line in lines if line)
    return description or None, image_links


def parse_contest_korea_html(
    html: str | bytes, base_url: str
) -> dict[str, str | list[str] | None]:
    # bytes를 전달받아 BeautifulSoup가 HTML의 charset 처리
    soup = BeautifulSoup(html, "html.parser")
    area = soup.select_one("div.view_top_area.clfx")
    if area is None:
        raise ValueError("공모전 정보 영역(div.view_top_area.clfx)을 찾을 수 없습니다.")
    text_area = area.select_one(".txt_area, .text_area")
    if text_area is None:
        raise ValueError(
            "공모전 테이블 영역(.txt_area 또는 .text_area)을 찾을 수 없습니다."
        )

    rows: dict[str, Tag] = {}
    for row in text_area.select("table tr"):
        heading, cell = row.find("th"), row.find("td")
        if heading is not None and cell is not None:
            rows[_label(_text(heading) or "")] = cell

    def field(*labels: str) -> Tag | None:
        for label in labels:
            cell = rows.get(_label(label))
            if cell is not None:
                return cell
        return None

    # 접수기간은 YYYY-MM-DD형식. 양 끝을 따로 처리해 '미정 ~ 2026.09.30'도 마감일로 인식
    period = _text(field("접수기간")) or ""
    endpoints = re.split(r"\s*[~～∼]\s*", period, maxsplit=1)
    application_start_at = _date(endpoints[0])
    application_end_at = _date(endpoints[1]) if len(endpoints) == 2 else None

    main_image_link = ""
    for img in area.select(".img_area img"):
        for attribute in ("data-src", "data-original", "src"):
            image_url = _absolute_url(img.get(attribute), base_url)
            if image_url:
                main_image_link = image_url
                break

    description, detail_image_links = _detail(soup, base_url, main_image_link)

    return {
        "title": _text(area.find("h1")) or _text(field("공모전명", "대회명")),
        "host_organization": _text(field("주최기관", "주최", "주최 . 주관")),
        # "host_organization_2": _text(field("주관기관", "주관", "주최 . 주관")),
        "category": _text(field("대표분야", "카테고리", "분야")),
        "target_participants": _text(field("참가대상")),
        "application_start_at": application_start_at,
        "application_end_at": application_end_at,
        "prize": _text(field("시상내역")),
        "homepage_link": _link(field("홈페이지"), base_url),
        "application_link": _link(field("접수하기"), base_url),
        "description": description,
        "main_image_link": main_image_link,
        "detail_image_links": detail_image_links,
    }


def parse_contest_korea(
    url: str, timeout: float = 30
) -> dict[str, str | list[str] | None]:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    response.raise_for_status()
    return parse_contest_korea_html(response.content, response.url)


if __name__ == "__main__":
    TEST_URLS = [
        "https://www.contestkorea.com/sub/view.php?int_gbn=1&Txt_bcode=031610001&str_no=202608190008",
        "https://www.contestkorea.com/sub/view.php?int_gbn=1&Txt_bcode=030210001&str_no=202608280052",
        "https://www.contestkorea.com/sub/view.php?int_gbn=1&Txt_bcode=031210001&str_no=202608240062",
        "https://www.contestkorea.com/sub/view.php?int_gbn=1&Txt_bcode=030210001&str_no=202608310026",
        # hard negatives
        "https://www.contestkorea.com/",
        "https://www.contestkorea.com/sub/list.php?displayrow=12&int_gbn=1&Txt_bcode=030510001",
    ]

    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("url", nargs="?", default=TEST_URL)
    # args = parser.parse_args()
    for url in TEST_URLS:
        try:
            print(
                json.dumps(
                    parse_contest_korea(url),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        except ValueError as e:
            print(e)
            continue
