from pathlib import Path

from app.retrieval_evaluation import evaluate_retrieval


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    summary = evaluate_retrieval(
        project_root / "data" / "processed" / "chunks.jsonl",
        project_root / "docs" / "evaluation_questions.jsonl",
    )

    assert summary.total == 14
    assert summary.answerable == 12
    assert 0 <= summary.recall_at_1 <= 1
    assert 0 <= summary.recall_at_3 <= 1
    assert 0 <= summary.section_hit_at_3 <= 1
    assert 0 <= summary.abstention_accuracy <= 1

    print("检索评测测试通过")


if __name__ == "__main__":
    main()
