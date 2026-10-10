from unittest.mock import patch

from app.errors import LLMClientError
from app.llm_client import chat, stream_chat


def main() -> None:
    with patch(
        "app.llm_client.client.chat.completions.create",
        side_effect=RuntimeError("simulated network failure"),
    ):
        try:
            chat("test request")
        except LLMClientError as exc:
            print(f"普通请求错误处理通过: {exc}")
        else:
            raise AssertionError("chat should raise LLMClientError")

    with patch(
        "app.llm_client.client.chat.completions.create",
        side_effect=RuntimeError("simulated stream failure"),
    ):
        try:
            list(stream_chat("test stream request"))
        except LLMClientError as exc:
            print(f"流式请求错误处理通过: {exc}")
        else:
            raise AssertionError("stream_chat should raise LLMClientError")

    print("错误处理测试通过")


if __name__ == "__main__":
    main()
