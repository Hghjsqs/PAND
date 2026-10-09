from app.llm_client import chat


if __name__ == "__main__":
    answer = chat("请只回答：API 调用成功")
    print(answer)