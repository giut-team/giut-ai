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


"""
response = client.chat.completions.create(
    model=settings.OPENAI_MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "이 이미지에 무엇이 있나요?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,..."},
                },
            ],
        }
    ],
)
"""
