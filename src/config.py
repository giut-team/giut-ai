import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    GPT_USAGE_LOG_DIR = os.getenv("GPT_USAGE_LOG_DIR", str(BASE_DIR / "logs"))
    X_AI_API_KEY = os.getenv("X_AI_API_KEY")


settings = Settings()
