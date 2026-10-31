import pytest

from src.parsers.contest_korea_parser import parse_contest_korea_html


@pytest.mark.parametrize(
    ("prize", "expected"),
    [
        ("상금(총상금 : 200만원 / 1위 : 50만원)", 2_000_000),
        ("상금(총 상금: 1,200만 원 / 1위 : 300만원)", 12_000_000),
        ("상금(1위 : 50만원)", None),
        ("상품(총 560만원 상당 시상품)", None),
    ],
)
def test_parse_contest_korea_total_prize_amount(prize, expected):
    html = f"""
        <div class="view_top_area clfx">
          <h1>대학생 문예 공모전</h1>
          <div class="txt_area">
            <table>
              <tr><th>주최기관</th><td>한국문예협회</td></tr>
              <tr><th>참가대상</th><td>대학생, 대학원생</td></tr>
              <tr><th>대표분야</th><td>문학•문예</td></tr>
              <tr><th>접수기간</th><td>2026.10.01 ~ 2026.10.31</td></tr>
              <tr><th>시상내역</th><td>{prize}</td></tr>
              <tr><th>홈페이지</th><td><a href="https://example.org">홈페이지</a></td></tr>
              <tr><th>접수하기</th><td><a href="https://example.org/apply">접수하기</a></td></tr>
            </table>
          </div>
        </div>
    """

    fields = parse_contest_korea_html(
        html, "https://www.contestkorea.com/sub/view.php", "contest"
    )

    assert fields["prize"] == prize
    assert fields["total_prize_amount"] == expected
