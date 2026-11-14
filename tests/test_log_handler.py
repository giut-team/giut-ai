from datetime import datetime
import json
import logging
from zoneinfo import ZoneInfo

import pytest

from src.utils import log_handler
from src.utils.log_handler import DailyRotatingFileHandler

SEOUL = ZoneInfo("Asia/Seoul")


@pytest.fixture
def request_log(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    test_logger = logging.Logger("test.backend_requests")
    monkeypatch.setattr(log_handler, "_request_logger", test_logger)
    monkeypatch.setattr(log_handler.settings, "BACKEND_REQUEST_LOG_DIR", str(log_dir))

    def read_records():
        path = log_dir / "backend_requests.json"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    try:
        yield read_records
    finally:
        for handler in test_logger.handlers[:]:
            test_logger.removeHandler(handler)
            handler.close()


class FakeResponse:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text

    def json(self):
        return json.loads(self.text)


def write_request(response=None, error=None):
    log_handler.write_backend_request_log(
        [{"title": "first"}],
        url="https://backend.example.com/competitions",
        sent_at=datetime(2026, 11, 14, 0, 0, 5, tzinfo=SEOUL),
        response=response,
        error=error,
    )


def test_request_log_stores_201_body_as_json(request_log):
    write_request(FakeResponse(201, '{"saved": [1, 2]}'))

    [record] = request_log()
    assert record == {
        "sent_at": "2026-11-14T00:00:05+09:00",
        "url": "https://backend.example.com/competitions",
        "count": 1,
        "status_code": 201,
        "response_body": {"saved": [1, 2]},
        "error": None,
        "payload": [{"title": "first"}],
    }


@pytest.mark.parametrize(
    ("status_code", "text"),
    [(400, '{"message": "bad request"}'), (201, "not json")],
)
def test_request_log_keeps_raw_text_for_non_201_or_invalid_json(
    request_log, status_code, text
):
    write_request(FakeResponse(status_code, text))

    [record] = request_log()
    assert record["status_code"] == status_code
    assert record["response_body"] == text


def test_request_log_records_error_without_response(request_log):
    write_request(error=ConnectionError("backend down"))

    [record] = request_log()
    assert record["status_code"] is None
    assert record["response_body"] is None
    assert record["error"] == "ConnectionError('backend down')"


def test_request_log_failure_does_not_raise(monkeypatch, request_log):
    def broken_ensure(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(log_handler, "_ensure_file_handler", broken_ensure)

    write_request(FakeResponse(201, "{}"))  # 예외가 밖으로 나오지 않아야 함


@pytest.fixture
def rotating(tmp_path):
    path = tmp_path / "app.json"
    handler = DailyRotatingFileHandler(path, tz=SEOUL, backup_count=2)
    handler.setFormatter(logging.Formatter("%(message)s"))
    # 첫 기록의 날짜를 파일 날짜로 고정
    handler._current_date = handler._next_date = datetime(2026, 11, 12).date()

    def emit_at(moment, message):
        record = logging.LogRecord("test", logging.INFO, "", 0, message, None, None)
        record.created = moment.timestamp()
        handler.handle(record)

    try:
        yield path, emit_at
    finally:
        handler.close()


def read_lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_rotation_uses_handler_timezone_date(rotating):
    path, emit_at = rotating
    emit_at(datetime(2026, 11, 12, 23, 59, tzinfo=SEOUL), "nov12")
    # 서울 11-13 00:01은 UTC로 11-12 15:01이지만 서울 날짜 기준으로 넘어가야 함
    emit_at(datetime(2026, 11, 13, 0, 1, tzinfo=SEOUL), "nov13-a")
    emit_at(datetime(2026, 11, 13, 8, 59, tzinfo=SEOUL), "nov13-b")

    assert read_lines(path.with_name("app.json.2026-11-12")) == ["nov12"]
    assert read_lines(path) == ["nov13-a", "nov13-b"]


def test_rotation_keeps_only_backup_count_files(rotating):
    path, emit_at = rotating
    for day in (12, 13, 14, 15):
        emit_at(datetime(2026, 11, day, 12, 0, tzinfo=SEOUL), f"nov{day}")

    assert sorted(p.name for p in path.parent.iterdir()) == [
        "app.json",
        "app.json.2026-11-13",
        "app.json.2026-11-14",
    ]
    assert read_lines(path) == ["nov15"]


def test_rotation_appends_to_existing_backup_of_same_date(rotating):
    path, emit_at = rotating
    path.with_name("app.json.2026-11-12").write_text("earlier\n", encoding="utf-8")

    emit_at(datetime(2026, 11, 12, 12, 0, tzinfo=SEOUL), "nov12")
    emit_at(datetime(2026, 11, 13, 12, 0, tzinfo=SEOUL), "nov13")

    assert read_lines(path.with_name("app.json.2026-11-12")) == ["earlier", "nov12"]
    assert read_lines(path) == ["nov13"]
