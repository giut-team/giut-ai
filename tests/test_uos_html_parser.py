from types import SimpleNamespace

import pytest

from src.parsers.uos_html_parser import _parse_uos_html, parse_uos_html


HTML = """
<html>
  <header><h3>사이트 메뉴</h3><img src="/logo.png"></header>
  <div class="view-bx">
    <div class="vw-tibx"><h3>[생활관] 수강생 모집</h3></div>
    <div class="vw-con">
      <p><span>2026</span><span>년 </span><strong>수강생</strong> 모집</p>
      <p>신청 안내<br>마감일 확인</p>
      <img src="/upload/poster.jpg">
      <img src="https://file.uos.ac.kr/other.png">
    </div>
  </div>
  <aside><img src="/unrelated.jpg"></aside>
</html>
"""


def test_parse_uos_html_extracts_only_notice_content():
    url = "https://www.uos.ac.kr/korNotice/view.do?list_id=FA1&seq=1"

    assert _parse_uos_html(HTML, url) == {
        "title": "[생활관] 수강생 모집",
        "link": url,
        "body": "2026년 수강생 모집\n신청 안내\n마감일 확인",
        "images": [
            "https://www.uos.ac.kr/upload/poster.jpg",
            "https://file.uos.ac.kr/other.png",
        ],
    }


def test_parse_uos_html_adds_anonymous_access_without_changing_link(monkeypatch):
    requested = []

    def fake_get(url, **kwargs):
        requested.append((url, kwargs))
        return SimpleNamespace(text=HTML, raise_for_status=lambda: None)

    monkeypatch.setattr("src.parsers.uos_html_parser.requests.get", fake_get)
    url = "https://www.uos.ac.kr/korNotice/view.do?list_id=FA1&seq=1"

    result = parse_uos_html(url)

    assert result["link"] == url
    assert requested[0][0] == url + "&identified=anonymous"


def test_parse_uos_html_rejects_non_notice_page():
    with pytest.raises(ValueError, match="공지 제목 또는 본문"):
        _parse_uos_html("<html><h3>로그인</h3></html>", "https://www.uos.ac.kr")
