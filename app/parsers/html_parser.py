from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urljoin

import requests
import urllib3
from bs4 import BeautifulSoup, Comment, NavigableString

# 최상위 컨테이너 (class/id 키워드 매칭에서 제외)
ROOT_TAGS = ("html", "body")

# 텍스트 블록 분리 기준 태그 목록 (인라인 태그 제외)
# a / img 등은 분리 기준으로 사용하지 않고 따로 뽑아 links/images 로 처리
BLOCK_TAGS = (
    "p",
    "div",
    "li",
    "dl",
    "dt",
    "dd",
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
)


# 본문에 포함되면 안 되는 노이즈 태그
NOISE_TAGS = (
    "head",
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "footer",
    "nav",
    "aside",
    "button",
)

# 광고/메뉴/배너 등 노이즈가 될 수 있는 class/id 키워드
NOISE_KEYWORDS = (
    "script",
    "nav",
    "aside",
    "button",
    "footer",
    "nav",
    "gnb",
    "lnb",
    "snb",
    "side",
    "menu",
    "sidebar",
    "ad",
    "ads",
    "advert",
    "popup",
    "modal",
    "cookie",
    "sns",
    "mail",
    "email",
    "mailing",
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

# href가 이미지로 바로 연결될 때 images로 분류할 확장자
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".ico")

# 접속 불가능한 스킴 (links에서 제외)
NON_NAVIGABLE_SCHEMES = ("javascript:", "mailto:", "tel:")

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; HtmlParserBot/1.0)"


@dataclass
class ParsedPage:
    # text / images / links 모두 (구조 블록 / 링크 / 이미지) 단위 feature dict 리스트
    text: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)

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
        # HTML -> 노이즈 제거 -> 남은 본문만 블록 단위로 분리 -> feature dict 생성
        # images, links 구할 때 상대경로(href/src)를 절대 URL로 변환하기 위해 base_url 필요

        soup = BeautifulSoup(content, features="html.parser")

        self._remove_noise(soup)

        root = soup.body or soup

        units = []
        for tag in root.find_all(BLOCK_TAGS + ("a", "img")):
            if tag.name in ("a", "img"):
                units.append(tag)
            elif self._own_text(tag):
                units.append(tag)
        total = len(units)

        text_blocks: list[dict] = []
        image_blocks: list[dict] = []
        link_blocks: list[dict] = []

        # 2) 각 블록을 feature dict 로 만든 뒤 img / a / 그 외(텍스트) 리스트로 분리
        for index, tag in enumerate(units):
            feat = self._block_features(tag, index, total)

            if tag.name == "img":
                src = self._img_src(tag)
                if not src:
                    continue
                feat["src"] = urljoin(base_url, src) if base_url else src
                image_blocks.append(feat)
            elif tag.name == "a":
                href = (tag.get("href") or "").strip()
                if not self._is_navigable(href):
                    continue
                feat["href"] = urljoin(base_url, href) if base_url else href
                link_blocks.append(feat)
            else:
                if not feat["text"]:
                    continue
                text_blocks.append(feat)

        return ParsedPage(text=text_blocks, images=image_blocks, links=link_blocks)

    def parse_url(self, url: str) -> ParsedPage:
        # fetch -> parse 바로 실행하는 헬퍼
        content = self.fetch(url)
        return self.parse(content, base_url=url)

    # -- internals -------------------------------------------------

    def _tokenize(self, *values: str) -> set[str]:
        text = " ".join(values).lower()
        return set(_TOKEN_SPLIT_RE.split(text)) - {""}

    def _normalize(self, content: str) -> str:
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

            tokens = self._tokenize(
                " ".join(tag.get("class", [])), tag.get("id", "") or ""
            )
            if tokens & NOISE_KEYWORD_SET:
                tag.decompose()

    def _ancestor_id(self, tag, depth=5) -> str:
        parts = []
        node = tag
        for _ in range(depth + 1):  # 자기 자신 포함
            if node is None or node.name in ("[document]", "html"):
                break
            ident = " ".join(node.get("class", []))
            if node.get("id"):
                ident += " #" + node["id"]
            parts.append(f"{node.name}:{ident.strip()}")
            node = node.parent
        return " > ".join(reversed(parts))

    def _block_features(self, tag, block_index: int, total_blocks: int) -> dict:
        # 이 블록 태그의 조상 id/class
        class_id = self._ancestor_id(tag)

        if tag.name == "img":
            block_text = self._normalize((tag.get("alt") or "").strip())
            links_in = []
        elif tag.name == "a":
            block_text = self._normalize(tag.get_text(" ", strip=True))
            links_in = []
        else:
            # 하위 블록의 텍스트/링크는 빼고 해당 블록의 직속 텍스트만 합치기
            block_text = self._own_text(tag)
            links_in = [a for a in tag.find_all("a") if self._is_own(tag, a)]

        link_char_count = sum(len(a.get_text(" ", strip=True)) for a in links_in)

        return {
            "text": block_text[:256],
            "tag": tag.name,
            "depth": len(list(tag.parents)),
            "text_len": len(block_text),
            "link_density": link_char_count / max(len(block_text), 1),
            "n_links": len(links_in),
            "rel_pos": block_index / max(total_blocks, 1),
            "class_id": class_id,
        }

    @staticmethod
    def _is_own(block, node) -> bool:
        # node(문자열 or 태그)와 block 사이에 다른 블록이 없으면 True
        parent = node.parent
        while parent is not None and parent is not block:
            if parent.name in BLOCK_TAGS:
                return False
            parent = parent.parent
        return parent is block

    def _own_text(self, block) -> str:
        # 블록의 직속 텍스트/인라인 요소만 모아주기 (중첩된 하위 블록 안의 텍스트는 제외)
        parts = [
            str(s)
            for s in block.descendants
            if isinstance(s, NavigableString)
            and not isinstance(s, Comment)
            and self._is_own(block, s)
        ]
        return self._normalize(" ".join(" ".join(parts).split()))

    @staticmethod
    def _img_src(img) -> str | None:
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if not src and img.get("srcset"):
            src = img["srcset"].split(",")[0].strip().split(" ")[0]
        return src.strip() if src else None

    @staticmethod
    def _is_navigable(href: str) -> bool:
        if not href or href.startswith("#"):
            return False
        low = href.lower()
        if low.startswith(NON_NAVIGABLE_SCHEMES):
            return False
        if low.split("?")[0].endswith(IMAGE_EXTENSIONS):
            # 이미지 링크는 images에서 이미 처리했으므로 제외
            return False
        return True


if __name__ == "__main__":
    import json
    import sys

    target_url = sys.argv[1] if len(sys.argv) > 1 else input("URL: ").strip()
    result = HtmlParser().parse_url(target_url)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print(
        f"text: {len(result.text)} / images: {len(result.images)} / links: {len(result.links)}"
    )
