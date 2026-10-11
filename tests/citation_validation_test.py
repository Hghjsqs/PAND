from app.structured_output import _validate_citation_targets
from app.retriever import SearchResult


def main() -> None:
    results = [
        SearchResult(
            chunk={
                "source": "leave_policy.md",
                "section": "2. 请假申请流程",
                "chunk_id": "leave-policy-test",
                "text": "请通过 OA 系统提交年假申请。",
            },
            score=1.0,
            matched_terms=("年", "假"),
        )
    ]

    _validate_citation_targets(
        [
            {
                "source": "leave_policy.md",
                "section": "2. 请假申请流程",
                "chunk_id": "leave-policy-test",
            }
        ],
        results,
    )

    try:
        _validate_citation_targets(
            [
                {
                    "source": "fake.md",
                    "section": "fake",
                    "chunk_id": "fake-id",
                }
            ],
            results,
        )
    except ValueError as exc:
        assert "does not match retrieved evidence" in str(exc)
    else:
        raise AssertionError("fake citation should be rejected")

    print("引用校验测试通过")


if __name__ == "__main__":
    main()
