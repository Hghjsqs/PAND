"""Read enterprise Markdown documents and produce deterministic chunks.

学习主线：原始文档不是直接拿去做向量化，而是先变成统一的
DocumentChunk。后面的 Embedding、BM25、引用和评测都依赖这个结构。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
METADATA_RE = {
    "document_id": re.compile(r"^文档编号[：:]\s*(.+?)\s*$"),
    "version": re.compile(r"^版本[：:]\s*(.+?)\s*$"),
    "effective_date": re.compile(r"^生效日期[：:]\s*(.+?)\s*$"),
    "department": re.compile(r"^负责部门[：:]\s*(.+?)\s*$"),
}


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable text unit with enough metadata for citation."""

    chunk_id: str
    source: str
    title: str
    document_id: str
    version: str
    effective_date: str
    department: str
    section: str
    text: str
    start_line: int
    end_line: int


def clean_markdown(text: str) -> str:
    """统一换行和空白，减少无意义字符对切块和检索的干扰。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _metadata(lines: list[str]) -> dict[str, str]:
    """读取文档头部的固定字段，不把它们混进正文。"""
    values = {"document_id": "", "version": "", "effective_date": "", "department": ""}
    for line in lines[:20]:
        for key, pattern in METADATA_RE.items():
            match = pattern.match(line.strip())
            if match:
                values[key] = match.group(1)
    return values


def _is_metadata_line(line: str) -> bool:
    """判断一行是否属于文档编号、版本等头部元数据。"""
    stripped = line.strip()
    return any(pattern.match(stripped) for pattern in METADATA_RE.values())


def _split_long_paragraph(text: str, max_chars: int, overlap: int) -> list[str]:
    """把超过长度上限的单个段落切开，并保留少量重叠内容。

    重叠可以减少答案刚好落在两个 Chunk 边界时的信息损失。
    """
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        pieces.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return pieces


def _section_chunks(
    heading: str,
    paragraphs: list[tuple[str, int, int]],
    max_chars: int,
    overlap: int,
) -> list[tuple[str, str, int, int]]:
    """把同一章节的多个段落合并成不超过 max_chars 的 Chunk。"""
    pieces: list[tuple[str, str, int, int]] = []
    current: list[str] = []
    current_start: int | None = None
    current_end: int | None = None

    def flush() -> None:
        # 把当前正在累积的段落写入结果，然后开始一个新 Chunk。
        nonlocal current, current_start, current_end
        if current and current_start is not None and current_end is not None:
            pieces.append((heading, "\n\n".join(current), current_start, current_end))
        current = []
        current_start = None
        current_end = None

    for paragraph, start_line, end_line in paragraphs:
        for part in _split_long_paragraph(paragraph, max_chars, overlap):
            candidate = "\n\n".join([*current, part]).strip()
            if current and len(candidate) > max_chars:
                # 加入当前段落会超长，因此先保存已有内容。
                flush()
            if current_start is None:
                current_start = start_line
            current_end = end_line
            current.append(part)
    flush()
    return pieces


def parse_markdown(markdown: str, source: str, max_chars: int = 700, overlap: int = 80) -> list[DocumentChunk]:
    """解析一个 Markdown 文件，并为每个 Chunk 保留可追溯信息。

    这里不调用模型，也不做向量化；职责只是把文本变成干净、稳定的
    数据结构。这样后续检索效果有问题时，可以先判断是切块问题还是模型问题。
    """
    if max_chars <= overlap or max_chars < 100:
        raise ValueError("max_chars must be greater than overlap and at least 100")

    cleaned = clean_markdown(markdown)
    lines = cleaned.splitlines()
    metadata = _metadata(lines)
    title = Path(source).stem
    sections: list[tuple[str, list[tuple[str, int, int]]]] = []
    current_heading = title
    current_paragraph: list[str] = []
    paragraph_start: int | None = None

    def flush_paragraph(end_line: int) -> None:
        nonlocal current_paragraph, paragraph_start
        paragraph = "\n".join(current_paragraph).strip()
        if paragraph and paragraph_start is not None:
            if sections and sections[-1][0] == current_heading:
                sections[-1][1].append((paragraph, paragraph_start, end_line))
            else:
                sections.append((current_heading, [(paragraph, paragraph_start, end_line)]))
        current_paragraph = []
        paragraph_start = None

    for line_number, line in enumerate(lines, start=1):
        heading_match = HEADING_RE.match(line)
        if heading_match:
            # 标题开始新的语义区域，后续正文归到这个章节。
            flush_paragraph(line_number - 1)
            heading = heading_match.group(2).strip()
            if heading_match.group(1) == "#":
                title = heading
                current_heading = heading
            else:
                current_heading = heading
            continue
        if _is_metadata_line(line):
            # 元数据已保存到字段，不应再次作为正文参与检索。
            flush_paragraph(line_number - 1)
            continue
        if not line.strip():
            flush_paragraph(line_number - 1)
            continue
        if paragraph_start is None:
            paragraph_start = line_number
        current_paragraph.append(line)
    flush_paragraph(len(lines))

    chunks: list[DocumentChunk] = []
    for heading, paragraphs in sections:
        for section, text, start_line, end_line in _section_chunks(
            heading, paragraphs, max_chars, overlap
        ):
            digest = hashlib.sha1(
                # 相同输入生成相同 ID，重复建库时不会产生随机 ID。
                f"{source}\0{section}\0{start_line}\0{end_line}\0{text}".encode("utf-8")
            ).hexdigest()[:12]
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{Path(source).stem}-{digest}",
                    source=source,
                    title=title,
                    document_id=metadata["document_id"],
                    version=metadata["version"],
                    effective_date=metadata["effective_date"],
                    department=metadata["department"],
                    section=section,
                    text=text,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
    return chunks


def load_markdown_documents(input_dir: Path) -> Iterable[DocumentChunk]:
    """按文件名排序读取目录中的 Markdown，保证处理顺序可复现。"""
    for path in sorted(input_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        yield from parse_markdown(path.read_text(encoding="utf-8"), path.name)


def write_jsonl(chunks: Iterable[DocumentChunk], output_path: Path) -> int:
    """把 Chunk 一行一个写入 JSONL，便于后续逐条读取和调试。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    # 命令行入口：读取 data/raw，输出 data/processed/chunks.jsonl。
    project_root = Path(__file__).resolve().parents[1]
    input_dir = project_root / "data" / "raw"
    output_path = project_root / "data" / "processed" / "chunks.jsonl"
    count = write_jsonl(load_markdown_documents(input_dir), output_path)
    print(f"processed_documents={sum(1 for _ in input_dir.glob('*.md') if _.name.lower() != 'readme.md')}")
    print(f"chunks={count}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
