"""Manual smoke test for streaming output.

Run from the project root with: python -m tests.stream_test
"""

from __future__ import annotations

import sys
import time

from app.llm_client import stream_chat


def main() -> None:
    started_at = time.perf_counter()
    first_fragment_at: float | None = None
    fragments: list[str] = []

    print("Streaming response:")
    for fragment in stream_chat(
        "Explain in three short sentences why an enterprise knowledge base needs citations."
    ):
        if first_fragment_at is None:
            first_fragment_at = time.perf_counter()
        fragments.append(fragment)
        sys.stdout.write(fragment)
        sys.stdout.flush()

    finished_at = time.perf_counter()
    answer = "".join(fragments)
    if not answer.strip():
        raise RuntimeError("streaming test produced an empty answer")

    print("\n\nMetrics:")
    print(f"first_fragment_ms={(first_fragment_at - started_at) * 1000:.0f}")
    print(f"total_ms={(finished_at - started_at) * 1000:.0f}")
    print(f"fragment_count={len(fragments)}")
    print(f"answer_chars={len(answer)}")


if __name__ == "__main__":
    main()
