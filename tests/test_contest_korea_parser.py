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
          <div class="txt_area">
            <table><tr><th>시상내역</th><td>{prize}</td></tr></table>
          </div>
        </div>
    """

    fields, _, _, _ = parse_contest_korea_html(
        html, "https://www.contestkorea.com/sub/view.php"
    )

    assert fields["prize"] == prize
    assert fields["total_prize_amount"] == expected
