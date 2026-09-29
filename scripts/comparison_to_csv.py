# LLM 테스트 결과 JSON의 comparison 항목을 일반 field / category CSV로 집계

"""
사용법:
    python app/utils/comparison_to_csv.py "tests/results/result.json"
    python app/utils/comparison_to_csv.py "result.json" --output-dir "reports"

- 기본 출력 위치: 입력 파일의 폴더
- 분모가 0인 지표는 빈 셀로 기록
- (category) F1 = 2×EXACT / (2×EXACT + ADDED + OMITTED)
- (category) EXACT, ADDED, OMITTED 개수가 모두 0이면 정답과 예측 모두 빈 것으로 간주, full_match와 both_empty에 포함
"""

import argparse
import csv
import json
from pathlib import Path

FIELD_COLUMNS = [
    "field",
    "total",
    "exact",
    "prize_normalized",
    "close",
    "exact_plus_close",
    "miss",
    "exact_rate",
    "exact_plus_close_rate",
]
CATEGORY_COLUMNS = [
    "field",
    "total",
    "exact",
    "added",
    "omitted",
    "precision",
    "recall",
    "f1",
    "full_match_count",
    "full_match_rate",
    "both_empty_count",
]


def percentage(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.2f}" if denominator else ""


def summarize_comparisons(records: list) -> tuple[list[dict], dict]:
    if not isinstance(records, list):
        raise ValueError("JSON 최상위 값은 문서 배열이어야 합니다.")

    fields = {}
    category_total = exact = added = omitted = full_match = both_empty = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(
            record.get("comparison"), dict
        ):
            raise ValueError(f"records[{index}].comparison은 객체여야 합니다.")

        for field, value in record["comparison"].items():
            location = f"records[{index}].comparison.{field}"
            if field == "category":
                if not isinstance(value, dict) or set(value) != {
                    "EXACT",
                    "ADDED",
                    "OMITTED",
                }:
                    raise ValueError(
                        f"{location}에는 EXACT, ADDED, OMITTED가 필요합니다."
                    )
                if any(type(count) is not int or count < 0 for count in value.values()):
                    raise ValueError(f"{location}의 개수는 0 이상의 정수여야 합니다.")
                category_total += 1
                exact += value["EXACT"]
                added += value["ADDED"]
                omitted += value["OMITTED"]
                if value["ADDED"] == 0 and value["OMITTED"] == 0:
                    full_match += 1
                    if value["EXACT"] == 0:
                        both_empty += 1
            else:
                if not isinstance(value, str) or value not in {
                    "EXACT",
                    "PRIZE_NORMALIZED",
                    "CLOSE",
                    "MISS",
                }:
                    raise ValueError(
                        f"{location}은 EXACT, PRIZE_NORMALIZED, CLOSE, MISS 중 하나여야 합니다."
                    )
                counts = fields.setdefault(
                    field, {"EXACT": 0, "PRIZE_NORMALIZED": 0, "CLOSE": 0, "MISS": 0}
                )
                counts[value] += 1

    field_rows = []
    for field, counts in fields.items():
        total = sum(counts.values())
        accepted = counts["EXACT"] + counts["PRIZE_NORMALIZED"] + counts["CLOSE"]
        field_rows.append(
            {
                "field": field,
                "total": total,
                "exact": counts["EXACT"],
                "prize_normalized": counts["PRIZE_NORMALIZED"],
                "close": counts["CLOSE"],
                "exact_plus_close": accepted,
                "miss": counts["MISS"],
                "exact_rate": percentage(counts["EXACT"], total),
                "exact_plus_close_rate": percentage(accepted, total),
            }
        )

    category_row = {
        "field": "category",
        "total": category_total,
        "exact": exact,
        "added": added,
        "omitted": omitted,
        "precision": percentage(exact, exact + added),
        "recall": percentage(exact, exact + omitted),
        "f1": percentage(2 * exact, 2 * exact + added + omitted),
        "full_match_count": full_match,
        "full_match_rate": percentage(full_match, category_total),
        "both_empty_count": both_empty,
    }
    return field_rows, category_row


def export_comparisons(
    input_path: str | Path, output_dir: str | Path | None = None
) -> tuple[Path, Path]:
    input_path = Path(input_path)
    with input_path.open(encoding="utf-8-sig") as source:
        field_rows, category_row = summarize_comparisons(json.load(source))

    destination = Path(output_dir) if output_dir is not None else input_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    fields_path = destination / "comparison_fields.csv"
    category_path = destination / "comparison_category.csv"
    for path, columns, rows in (
        (fields_path, FIELD_COLUMNS, field_rows),
        (category_path, CATEGORY_COLUMNS, [category_row]),
    ):
        # Excel에서도 한글을 읽을 수 있도록 UTF-8 BOM 포함
        with path.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return fields_path, category_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path, help="LLM 결과 JSON 파일")
    parser.add_argument(
        "--output-dir", type=Path, help="출력 폴더 (기본: 입력 파일 폴더)"
    )
    args = parser.parse_args()
    try:
        paths = export_comparisons(args.input_json, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"오류: {error}\n")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
