# RAG 问答基线

## 这一步解决什么问题

关键词检索只能找到原文，还不能把原文组织成自然的回答。RAG（Retrieval-Augmented Generation，检索增强生成）把流程拆成两步：

```text
问题 -> 检索相关 Chunk -> 把 Chunk 放进 Prompt -> Qwen 根据证据回答
```

模型不直接读取整个知识库，只读取本次问题检索出的少量证据，因此答案更容易追溯，也能减少无关内容干扰。

## 文件职责

- `app/retriever.py`：从 JSONL 中检索相关 Chunk。
- `app/rag.py`：格式化证据、构造 Prompt、调用 Qwen。
- `tests/rag_test.py`：离线检查 Prompt 是否包含问题、正文、来源和章节。

## 运行离线测试

```powershell
python -m tests.rag_test
```

## 调用真实模型

先重新生成 Chunk，再运行：

```powershell
python -m app.document_processor
python -m app.rag
```

输入 `年假怎么申请` 后，模型应只依据检索到的制度原文回答，并在关键结论后标注来源。
