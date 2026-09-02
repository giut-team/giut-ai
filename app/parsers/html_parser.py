from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urljoin

import requests
import trafilatura
import urllib3
from bs4 import BeautifulSoup

# 최상위 컨테이너 (class/id 키워드 매칭에서 제외)
ROOT_TAGS = ("html", "body")

# trafilatura XML 출력의 블록 레벨 태그 (인라인 <ref> 는 제외)
TRAFILATURA_BLOCK_TAGS = (
    "p",
    "head",
    "item",
    "row",
    "cell",
    "quote",
    "code",
    "list",
    "table",
    "lb",
    "graphic",
)

# 줄바꿈 기준 태그 목록 (인라인 태그 제외)
BLOCK_TAGS = (
    "p",
    "div",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "tr",
    "table",
    "section",
    "article",
    "blockquote",
    "pre",
    "br",
)

# 본문에 포함되면 안 되는 노이즈 태그
NOISE_TAGS = (
    "head",
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "header",
    "footer",
    "nav",
    "aside",
    "button",
)

# 광고/메뉴/배너 등 노이즈가 될 수 있는 class/id 키워드
NOISE_KEYWORDS = (
    "script",
    "header",
    "footer",
    "nav",
    "aside",
    "button",
    "footer",
    "nav",
    "gnb",
    "lnb",
    "menu",
    "sidebar",
    "banner",
    "ad",
    "ads",
    "advert",
    "popup",
    "modal",
    "cookie",
    "sns",
    "share",
    "comment",
    "related",
    "breadcrumb",
    "pagination",
)

NOISE_KEYWORD_SET = frozenset(NOISE_KEYWORDS)

# class/id를 영숫자 구간 단위로 나누기
# ex) "fade"가 "ad"를 부분 문자열로 포함해 필터링하지 않도록
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _tokenize(*values: str) -> set[str]:
    text = " ".join(values).lower()
    return set(_TOKEN_SPLIT_RE.split(text)) - {""}


# href가 이미지로 바로 연결될 때 images로 분류할 확장자
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".ico")

# 접속 불가능한 스킴 (links에서 제외)
NON_NAVIGABLE_SCHEMES = ("javascript:", "mailto:", "tel:")

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; HtmlParserBot/1.0)"


@dataclass
class ParsedPage:
    text: str
    images: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"text": self.text, "images": self.images, "links": self.links}


class HtmlParser:
    def __init__(self, timeout: int = 10, user_agent: str = DEFAULT_USER_AGENT):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def fetch(self, url: str) -> bytes:
        # url 요청 -> HTML 원문(bytes)을 반환
        # parse()의 BeautifulSoup에서 디코딩 (<meta charset> / BOM / 헤더 종합 판단)
        try:
            response = self.session.get(url, timeout=self.timeout)
        except requests.exceptions.SSLError:
            # 인증서 검증 실패 시 검증 없이 재시도
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = self.session.get(url, timeout=self.timeout, verify=False)

        response.raise_for_status()
        return response.content

    def parse(self, content: str | bytes, base_url: str = "") -> ParsedPage:
        # HTML 문자열을 text / images / links로 분리
        # images, links 구할 때 상대경로(href/src)를 절대 URL로 변환하기 위해 base_url 필요

        # 1차: trafilatura로 본문(중요 영역)만 추출 -> 그 영역 안의 text/images/links만 사용
        page = self._parse_with_trafilatura(content, base_url)
        if page is not None and page.text:
            return page

        # 2차(폴백): trafilatura가 본문을 못 찾으면 기존 BeautifulSoup 파이프라인 사용
        return self._parse_with_soup(content, base_url)

    def parse_url(self, url: str) -> ParsedPage:
        # fetch -> parse 바로 실행하는 헬퍼
        content = self.fetch(url)
        return self.parse(content, base_url=url)

    # -- internals -------------------------------------------------

    def _parse_with_trafilatura(
        self, content: str | bytes, base_url: str
    ) -> ParsedPage | None:
        # trafilatura가 판단한 본문 영역을 XML로 받아 내부 텍스트/이미지/링크만 추출 -> 메뉴, 관련 공모전 목록, 사이드바 등 본문 밖 요소는 애초에 들어오지 않음

        # nav/footer/모달 등 노이즈 우선 제거 후 그 안에서 본문 판별
        pre_soup = BeautifulSoup(content, features="html.parser")
        self._remove_noise(pre_soup)

        xml = trafilatura.extract(
            str(pre_soup),
            url=base_url or None,
            output_format="xml",
            include_comments=False,
            include_images=True,
            include_links=True,
            include_tables=True,
            favor_recall=True,
            deduplicate=True,
        )
        if not xml:
            return None

        soup = BeautifulSoup(xml, features="html.parser")
        main = soup.find("main") or soup

        images: list[str] = []
        seen_images: set[str] = set()
        for graphic in main.find_all("graphic"):
            self._add_url(images, seen_images, graphic.get("src"), base_url)

        links: list[str] = []
        seen_links: set[str] = set()
        for ref in main.find_all("ref"):
            href = (ref.get("target") or "").strip()
            if not href or href.startswith("#"):
                continue
            if href.lower().startswith(NON_NAVIGABLE_SCHEMES):
                continue
            if href.lower().split("?")[0].endswith(IMAGE_EXTENSIONS):
                # 이미지 링크는 <graphic>에서 이미 처리함
                continue
            self._add_url(links, seen_links, href, base_url)

        # 블록 레벨 태그 뒤에만 개행 (인라인 <ref> 텍스트는 문장 안에 유지)
        for tag in main.find_all(TRAFILATURA_BLOCK_TAGS):
            tag.insert_after("\n")

        lines = [
            self._normalize(line.strip())
            for line in main.get_text().splitlines()
            if line.strip()
        ]
        return ParsedPage(text="\n".join(lines), images=images, links=links)

    def _parse_with_soup(self, content: str | bytes, base_url: str) -> ParsedPage:
        soup = BeautifulSoup(content, features="html.parser")

        # 노이즈 제거 후 남은 본문에서만 이미지/링크/텍스트 추출
        self._remove_noise(soup)

        images = self._extract_images(soup, base_url)
        links = self._extract_links(soup, base_url)
        text = self._extract_text(soup)

        return ParsedPage(text=text, images=images, links=links)

    def _normalize(self, content: str):
        normalized = re.sub(
            r"[\u0000\u200B\uFEFF]+", "", content
        )  # 폭이 0인 특수문자 제거
        normalized = re.sub(
            r"[\t\r\u00A0]+", " ", normalized
        )  # \t, \r, &nbsp; 등이 하나 또는 연속으로 있을 때 스페이스 1개로 처리
        return normalized

    def _remove_noise(self, soup: BeautifulSoup) -> None:
        for tag in soup.find_all(True):
            if getattr(tag, "decomposed", False):
                continue

            if tag.name in NOISE_TAGS:
                tag.decompose()
                continue

            if tag.name in ROOT_TAGS:
                # <html>/<body>의 상태 클래스에 노이즈 키워드가 우연히 섞여 decompose하는 경우 방지
                # ex) "mobile-nav-on"에서 "nav"
                continue

            tokens = _tokenize(" ".join(tag.get("class", [])), tag.get("id", "") or "")
            if tokens & NOISE_KEYWORD_SET:
                tag.decompose()

    def _extract_text(self, soup: BeautifulSoup) -> str:
        # 블록 레벨 태그 뒤에만 명시적으로 개행
        for tag in soup.find_all(BLOCK_TAGS):
            tag.insert_after("\n")

        lines = [
            self._normalize(line.strip())
            for line in soup.get_text().splitlines()
            if line.strip()
        ]
        return "\n".join(lines)

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()

        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-original")
            if not src and img.get("srcset"):
                src = img["srcset"].split(",")[0].strip().split(" ")[0]
            self._add_url(urls, seen, src, base_url)

        for source in soup.find_all("source"):
            if source.get("srcset"):
                src = source["srcset"].split(",")[0].strip().split(" ")[0]
                self._add_url(urls, seen, src, base_url)

        return urls

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("#"):
                continue
            if href.lower().startswith(NON_NAVIGABLE_SCHEMES):
                continue
            if href.lower().split("?")[0].endswith(IMAGE_EXTENSIONS):
                # 이미지 링크는 _extract_images()에서 이미 처리했으므로 넘어가기
                continue
            self._add_url(urls, seen, href, base_url)

        return urls

    @staticmethod
    def _add_url(
        urls: list[str], seen: set[str], value: str | None, base_url: str
    ) -> None:
        if not value:
            return
        value = value.strip()
        if not value:
            return
        resolved = urljoin(base_url, value) if base_url else value
        if resolved not in seen:
            seen.add(resolved)
            urls.append(resolved)


if __name__ == "__main__":
    import json
    import sys

    target_url = sys.argv[1] if len(sys.argv) > 1 else input("URL: ").strip()
    result = HtmlParser().parse_url(target_url)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
