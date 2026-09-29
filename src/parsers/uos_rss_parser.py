# 서울시립대학교 RSS에서 제목, 원문 링크, 본문 텍스트, 게시일, 이미지 src 추출

import json

import feedparser
import requests
from bs4 import BeautifulSoup

RSS_URL_GENERAL = "https://www.uos.ac.kr/rss/gBoard.do"
RSS_URL_HAKSA = "https://www.uos.ac.kr/rss/hBoard.do"


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    lines = (
        " ".join(line.split()) for line in soup.get_text(separator="\n").splitlines()
    )
    return "\n".join(line for line in lines if line)


def _extract_image_sources(html: str) -> list[str]:
    # 본문 내 <img> 태그에서 src 링크 모두 추출
    soup = BeautifulSoup(html, "html.parser")
    return [
        src for image in soup.find_all("img", src=True) if (src := image["src"].strip())
    ]


def parse_uos_rss(
    url: str = RSS_URL_GENERAL, timeout: float = 30
) -> list[dict[str, str | list[str]]]:
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        },
        timeout=timeout,
    )
    response.raise_for_status()

    # 바이트를 전달하면 feedparser가 XML의 문자 인코딩 처리
    feed = feedparser.parse(response.content)
    if not feed.version or feed.bozo:
        raise ValueError(f"RSS 피드 파싱 오류: {feed.get('bozo_exception', url)}")

    return [
        {
            "title": entry.get("title", "").strip(),
            "link": entry.get("link", "").strip(),
            "body": _html_to_text(entry.get("summary", "")),
            "pubDate": entry.get("published", "").strip(),
            "images": _extract_image_sources(entry.get("summary", "")),
        }
        for entry in feed.entries
    ]


if __name__ == "__main__":
    print(json.dumps(parse_uos_rss(), ensure_ascii=False, indent=2))
