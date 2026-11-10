import asyncio
from datetime import datetime
import json
import logging
from threading import Barrier

from openai.types import CompletionUsage
import pytest


@pytest.fixture
def usage_log(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from src.llm import field_extractor

    log_dir = tmp_path / "logs"
    test_logger = logging.Logger("test.gpt_usage")
    monkeypatch.setattr(field_extractor, "_usage_logger", test_logger)
    monkeypatch.setattr(field_extractor.settings, "GPT_USAGE_LOG_DIR", str(log_dir))
    arguments = {
        "source_url": "https://example.com/공모전",
        "request_type": "text",
        "run_id": "run-001",
        "model": "test-model",
        "started_at": datetime.fromisoformat("2026-11-08T00:00:00+09:00"),
        "ended_at": datetime.fromisoformat("2026-11-08T00:00:01.234567+09:00"),
    }
    try:
        yield field_extractor.write_gpt_usage_log, arguments, log_dir, test_logger
    finally:
        for handler in test_logger.handlers[:]:
            test_logger.removeHandler(handler)
            handler.close()


def test_usage_log_preserves_timezone_duration_and_token_details(usage_log):
    write_log, arguments, log_dir, _ = usage_log
    usage = CompletionUsage(
        prompt_tokens=100,
        completion_tokens=30,
        total_tokens=130,
        prompt_tokens_details={"cached_tokens": 60},
        completion_tokens_details={"reasoning_tokens": 20},
    )
    write_log(usage, **arguments, response_id="response-001")

    lines = (log_dir / "gpt_usage.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["source_url"] == arguments["source_url"]
    assert record["started_at"] == "2026-11-07T15:00:00.000+00:00"
    assert record["ended_at"] == "2026-11-07T15:00:01.234+00:00"
    assert record["elapsed_ms"] == 1234.567
    assert record["input_tokens"] == 100
    assert record["cached_input_tokens"] == 60
    assert record["output_tokens"] == 30
    assert record["reasoning_tokens"] == 20
    assert record["response_id"] == "response-001"
    assert "cache_rate" not in record
    assert "elapsed_time" not in record


@pytest.mark.parametrize(
    ("ended_at", "expected_ms"),
    [
        ("2026-11-07T15:00:01.234567+00:00", 1234.567),
        ("2026-11-08T00:00:00+09:00", 0),
    ],
)
def test_elapsed_ms_handles_different_timezones_and_zero_duration(
    usage_log, ended_at, expected_ms
):
    write_log, arguments, log_dir, _ = usage_log
    write_log(None, **{**arguments, "ended_at": datetime.fromisoformat(ended_at)})
    record = json.loads((log_dir / "gpt_usage.jsonl").read_text(encoding="utf-8"))
    assert record["elapsed_ms"] == expected_ms


def test_usage_log_records_unknown_usage_as_null(usage_log):
    write_log, arguments, log_dir, _ = usage_log
    write_log(None, **arguments, status="error", error_type="APITimeoutError")
    write_log(
        CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        **{**arguments, "request_type": "image"},
    )

    error, image = [
        json.loads(line)
        for line in (log_dir / "gpt_usage.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert error["status"] == "error"
    assert error["error_type"] == "APITimeoutError"
    for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"):
        assert error[key] is None
    assert image["run_id"] == error["run_id"]
    assert image["type"] == "image"
    assert image["input_tokens"] == 0
    assert image["cached_input_tokens"] is None
    assert image["reasoning_tokens"] is None


def test_concurrent_calls_share_one_handler_without_missing_or_duplicate_lines(usage_log):
    write_log, arguments, log_dir, test_logger = usage_log
    barrier = Barrier(5)

    def write_one(index):
        barrier.wait(timeout=5)
        write_log(
            CompletionUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
            **{**arguments, "run_id": f"run-{index}"},
        )

    async def write_all():
        await asyncio.gather(*(asyncio.to_thread(write_one, index) for index in range(5)))

    asyncio.run(write_all())

    records = [
        json.loads(line)
        for line in (log_dir / "gpt_usage.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 5
    assert {record["run_id"] for record in records} == {f"run-{i}" for i in range(5)}
    assert len(test_logger.handlers) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"started_at": datetime(2026, 11, 8)},
        {"ended_at": datetime(2026, 11, 8)},
        {"request_type": "other"},
        {"ended_at": datetime.fromisoformat("2026-11-07T14:59:59+00:00")},
    ],
)
def test_invalid_usage_log_is_rejected_before_creating_a_file(usage_log, overrides):
    write_log, arguments, log_dir, _ = usage_log
    with pytest.raises(ValueError):
        write_log(None, **{**arguments, **overrides})
    assert not log_dir.exists()
