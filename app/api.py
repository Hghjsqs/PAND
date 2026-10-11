"""HTTP API：把企业知识库问答能力暴露给前端或其他服务。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.errors import LLMClientError
from app.structured_output import classify_question_stream_result


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

app = FastAPI(title="Enterprise Knowledge Base QA", version="0.1.0")


@app.middleware("http")
async def ensure_json_utf8(request: Request, call_next):
    """明确声明 JSON 使用 UTF-8，避免 Windows PowerShell 误判响应编码。"""

    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json") and "charset=" not in content_type:
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response


class ChatRequest(BaseModel):
    """客户端提交的问题。"""

    question: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=3, ge=1, le=10)


class Citation(BaseModel):
    """回答引用的知识库 Chunk。"""

    source: str
    section: str
    chunk_id: str


class ChatResponse(BaseModel):
    """结构化问答结果。"""

    answer: str
    category: str
    need_retrieval: bool
    confidence: float
    citations: list[Citation]


@app.get("/health")
def health() -> dict[str, str]:
    """确认服务进程是否正常运行。"""

    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> dict[str, Any]:
    """检索知识库并返回一个结构化答案。"""

    try:
        return classify_question_stream_result(
            request.question,
            chunks_path=CHUNKS_PATH,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
