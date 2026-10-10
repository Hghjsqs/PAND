from collections.abc import Iterator

from openai import OpenAI

from app.config import DASHSCOPE_API_KEY, QWEN_BASE_URL, QWEN_MODEL


client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=QWEN_BASE_URL,
    timeout=30.0,
    max_retries=1,
)


def chat(user_prompt: str) -> str:
    """向 Qwen 发送一次最小对话请求。"""
    if not user_prompt.strip():
        raise ValueError("user_prompt cannot be empty")

    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一个简洁、准确的企业知识库助手。",
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
    )

    if not response.choices:
        raise RuntimeError("model returned no choices")

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("model returned empty content")

    return content


def stream_chat(user_prompt: str) -> Iterator[str]:
    """Yield generated text fragments as Qwen produces them."""
    if not user_prompt.strip():
        raise ValueError("user_prompt cannot be empty")

    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a concise and accurate enterprise knowledge-base assistant.",
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
        stream=True,
    )

    yielded = False
    for chunk in response:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            yielded = True
            yield content

    if not yielded:
        raise RuntimeError("model returned no streamed content")
