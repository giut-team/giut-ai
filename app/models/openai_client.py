from openai import AsyncOpenAI
from app.config import settings

client = AsyncOpenAI(settings.OPENAI_API_KEY)


def create_json_completion(messages: list[dict], response_format: type):
    return client.chat.completions.parse(
        model=settings.OPENAI_MODEL,
        messages=messages,
        response_format=response_format,
        store=False,
        temperature=0.1,
    )
