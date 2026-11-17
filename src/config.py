import os
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")

    TEXT_CACHE_KEY = "giut-ai-text-extraction"
    IMAGE_CACHE_KEY = "giut-ai-image-extraction"
    UOS_TEXT_CACHE_KEY = "giut-ai-uos-text-extraction"
    UOS_IMAGE_CACHE_KEY = "giut-ai-uos-image-extraction"

    GPT_USAGE_LOG_DIR = os.getenv(
        "GPT_USAGE_LOG_DIR", str(BASE_DIR / "tmp" / "logs" / "gpt_usage")
    )
    BACKEND_REQUEST_LOG_DIR = os.getenv(
        "BACKEND_REQUEST_LOG_DIR", str(BASE_DIR / "tmp" / "logs")
    )
    API_REQUEST_LOG_DIR = os.getenv(
        "API_REQUEST_LOG_DIR", str(BASE_DIR / "tmp" / "logs" / "api_requests")
    )

    COLLECTOR_STATE_PATH = os.getenv(
        "COLLECTOR_STATE_PATH", str(BASE_DIR / "tmp" / "collector_state.json")
    )

    X_AI_API_KEY = os.getenv("X_AI_API_KEY")
    COMPETITIONS_API_URL = os.getenv("COMPETITIONS_API_URL")

    TIMEZONE = ZoneInfo("Asia/Seoul")


settings = Settings()
