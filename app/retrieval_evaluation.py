"""Evaluate retrieval quality against the project's labeled questions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from app.retriever import SearchResult, search_jsonl


@dataclass(frozen=True)
class EvaluationCase:
    """One labeled question from the evaluation JSONL file."""

    question_id: str
    question: str
    source: str | None
    section: str | None
    question_type: str


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate retrieval metrics for one evaluation run."""

    total: int
    answerable: int
    recall_at_1: float
    recall_at_3: float
    section_hit_at_3: float
    abstention_accuracy: float


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    """Read labeled questions without loading their expected answers into retrieval."""

    cases: list[EvaluationCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid evaluation JSON on line {line_number}") from exc
            required = {"id", "question", "source", "section", "type"}
            if not required.issubset(value):
                missing = sorted(required - value.keys())
                raise ValueError(f"evaluation case missing fields: {missing}")
            cases.append(
                EvaluationCase(
                    question_id=str(value["id"]),
                    question=str(value["question"]),
                    source=value["source"],
                    section=value["section"],
                    question_type=str(value["type"]),
                )
            )
    return cases


def _source_hit(results: list[SearchResult], expected_source: str | None) -> bool:
    return expected_source is not None and any(
        result.chunk.get("source") == expected_source for result in results
    )


def _section_hit(results: list[SearchResult], case: EvaluationCase) -> bool:
    return case.section is not None and any(
        result.chunk.get("source") == case.source
        and result.chunk.get("section") == case.section
        for result in results
    )


def evaluate_retrieval(
    chunks_path: Path,
    questions_path: Path,
    top_k: int = 3,
) -> EvaluationSummary:
    """Run retrieval against every labeled question and calculate aggregate metrics."""

    if top_k < 3:
        raise ValueError("top_k must be at least 3 for the standard evaluation")

    cases = load_evaluation_cases(questions_path)
    answerable = [case for case in cases if case.source is not None]
    recall_1_hits = 0
    recall_3_hits = 0
    section_hits = 0
    abstention_hits = 0

    for case in cases:
        results = search_jsonl(chunks_path, case.question, top_k=top_k)
        top_one = results[:1]
        if case.source is None:
            # 无答案/范围外问题的理想检索结果是空，避免把无关 Chunk 交给模型。
            abstention_hits += int(not results)
            continue
        recall_1_hits += int(_source_hit(top_one, case.source))
        recall_3_hits += int(_source_hit(results[:3], case.source))
        section_hits += int(_section_hit(results[:3], case))

    answerable_count = len(answerable)
    abstention_count = len(cases) - answerable_count
    return EvaluationSummary(
        total=len(cases),
        answerable=answerable_count,
        recall_at_1=recall_1_hits / answerable_count if answerable_count else 0.0,
        recall_at_3=recall_3_hits / answerable_count if answerable_count else 0.0,
        section_hit_at_3=section_hits / answerable_count if answerable_count else 0.0,
        abstention_accuracy=abstention_hits / abstention_count if abstention_count else 0.0,
    )


def main() -> None:
    """Print the lexical retriever baseline metrics."""

    project_root = Path(__file__).resolve().parents[1]
    summary = evaluate_retrieval(
        project_root / "data" / "processed" / "chunks.jsonl",
        project_root / "docs" / "evaluation_questions.jsonl",
    )
    print(f"total={summary.total}")
    print(f"answerable={summary.answerable}")
    print(f"recall_at_1={summary.recall_at_1:.3f}")
    print(f"recall_at_3={summary.recall_at_3:.3f}")
    print(f"section_hit_at_3={summary.section_hit_at_3:.3f}")
    print(f"abstention_accuracy={summary.abstention_accuracy:.3f}")


if __name__ == "__main__":
    main()
