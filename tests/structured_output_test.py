from app.structured_output import classify_question_stream_result


QUESTIONS = [
    "年假怎么申请？",
    "电脑无法连接公司网络怎么办？",
    "公司报销需要哪些材料？",
    "今天天气怎么样？",
]


def print_fragment(fragment: str) -> None:
    """Print each model fragment immediately as it arrives."""
    print(fragment, end="", flush=True)


def main() -> None:
    for question in QUESTIONS:
        print(f"问题：{question}")
        print("流式回答：", end="", flush=True)

        result = classify_question_stream_result(
            question,
            on_fragment=print_fragment,
        )

        print("\n流式回答结束")
        print(f"结构化结果：{result}\n")

    try:
        classify_question_stream_result("   ")
    except ValueError as exc:
        print(f"空问题校验通过：{exc}")
    else:
        raise AssertionError("empty question should raise ValueError")


if __name__ == "__main__":
    main()
