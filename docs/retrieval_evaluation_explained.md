# 检索评测说明

## 为什么需要指标

单个问题能找到正确文档，不代表整体检索可靠。评测集为每道问题标注了标准来源文件和章节，系统可以用它们衡量检索结果。

## 指标含义

- `Recall@1`：正确来源是否排在第一名。
- `Recall@3`：正确来源是否出现在前三名。
- `Section Hit@3`：正确文件和正确章节是否出现在前三名。
- `abstention_accuracy`：无答案或范围外问题是否保持空检索结果。

## 运行

```powershell
python -m app.document_processor
python -m app.retrieval_evaluation
python -m tests.retrieval_evaluation_test
```

这些指标是后续替换 Embedding、BM25 或混合检索时的对照基线。改进检索方案后，应使用同一批问题重新评测，不能只挑几个成功案例。
