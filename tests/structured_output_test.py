from app.structured_output import classify_question


QUESTIONS = [
    "年假怎么申请？",
    "电脑无法连接公司网络怎么办？",
    "公司报销需要哪些材料？",
    "今天天气怎么样？",
]


def main() -> None:
    for question in QUESTIONS:
        result = classify_question(question)
        print(f"问题: {question}")
        print(f"结果: {result}")
        print()

    try:
        classify_question("   ")
    except ValueError as exc:
        print(f"空问题校验通过: {exc}")
    else:
        raise AssertionError("empty question should raise ValueError")


if __name__ == "__main__":
    main()
