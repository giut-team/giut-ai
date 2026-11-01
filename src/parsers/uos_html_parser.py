# 시립대 개별 공지의 제목, 본문, 이미지 추출

from urllib.parse import parse_qs, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

from src.parsers.uos_rss_parser import _html_to_text


def _parse_uos_html(html: str, url: str) -> dict[str, str | list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    title_node = soup.select_one(".view-bx .vw-tibx h3")
    body_node = soup.select_one(".view-bx .vw-con")
    if title_node is None or body_node is None:
        raise ValueError(
            f"서울시립대학교 공지 제목 또는 본문을 찾을 수 없습니다: {url}"
        )

    images = [
        urljoin(url, src)
        for image in body_node.select("img[src]")
        if (src := image["src"].strip())
    ]
    return {
        "title": " ".join(title_node.get_text().split()),
        "link": url,
        "body": _html_to_text(str(body_node)),
        "images": images,
    }


def parse_uos_html(url: str, timeout: float = 30) -> dict[str, str | list[str]]:
    # 개별 공지 URL 요청 및 제목, 본문, 이미지 추출
    url = url.strip()
    parts = urlsplit(url)
    query = parts.query
    if (
        parts.hostname == "www.uos.ac.kr"
        and parts.path == "/korNotice/view.do"
        and "identified" not in parse_qs(query)
    ):
        query = f"{query}&identified=anonymous" if query else "identified=anonymous"
    fetch_url = urlunsplit(parts._replace(query=query, fragment=""))

    response = requests.get(
        fetch_url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    return _parse_uos_html(response.text, url)


if __name__ == "__main__":
    result = parse_uos_html(
        "https://www.uos.ac.kr/korNotice/view.do?list_id=FA1&seq=31636&sort=1"
    )
    for key, value in result.items():
        print(f"{key}: {value}")
