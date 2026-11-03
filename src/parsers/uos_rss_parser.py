# 서울시립대학교 RSS에서 제목, 원문 링크, 본문 텍스트, 게시일, 이미지 src 추출

import json
import re

import feedparser
import requests
from bs4 import BeautifulSoup, NavigableString, Tag


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    block_tags = {
        "p",
        "div",
        "section",
        "article",
        "li",
        "ul",
        "ol",
        "blockquote",
        "table",
    }
    block_tags.update(f"h{level}" for level in range(1, 7))

    def render(node: Tag | NavigableString) -> str:
        if isinstance(node, NavigableString):
            # HTML 소스의 들여쓰기/줄바꿈은 문장 안에서는 공백으로 취급
            return re.sub(r"\s+", " ", str(node))
        if node.name in {"script", "style", "noscript"}:
            return ""
        if node.name == "br":
            return "\n"
        if node.name == "tr":
            cells = node.find_all(["th", "td"], recursive=False)
            if cells:
                contents = " | ".join(" ".join(render(cell).split()) for cell in cells)
            else:
                contents = "".join(render(child) for child in node.children)
            return f"\n{contents}\n"

        contents = "".join(render(child) for child in node.children)
        return f"\n{contents}\n" if node.name in block_tags else contents

    lines = (" ".join(line.split()) for line in render(soup).splitlines())
    return "\n".join(line for line in lines if line)


def _extract_image_sources(html: str) -> list[str]:
    # 본문 내 <img> 태그에서 src 링크 모두 추출
    soup = BeautifulSoup(html, "html.parser")
    return [
        src for image in soup.find_all("img", src=True) if (src := image["src"].strip())
    ]


def parse_uos_rss(url: str, timeout: float = 30) -> list[dict[str, str | list[str]]]:
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
            "published_date": entry.get("published", "").strip(),
            "images": _extract_image_sources(entry.get("summary", "")),
        }
        for entry in feed.entries
    ]


if __name__ == "__main__":
    print(
        json.dumps(
            parse_uos_rss("https://www.uos.ac.kr/rss/gBoard.do"),
            ensure_ascii=False,
            indent=2,
        )
    )
