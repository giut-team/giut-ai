from openai import AsyncOpenAI
from src.config import settings

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


def create_json_completion(
    messages: list[dict], response_format: type, prompt_cache_key: str
):
    return client.chat.completions.parse(
        model=settings.OPENAI_MODEL,
        messages=messages,
        response_format=response_format,
        store=False,
        timeout=120.0,
        reasoning_effort="low",
        prompt_cache_key=prompt_cache_key,
    )
