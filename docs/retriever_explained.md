# 检索基线说明

## 目标

大模型回答企业制度问题前，需要先从 `data/processed/chunks.jsonl` 找到相关原文。
本阶段先实现一个不依赖第三方服务的关键词检索器，方便观察每个分数和命中的词。

## 工作流程

```text
用户问题
  -> 分词
  -> 与每个 Chunk 的词集合求交集
  -> 按命中比例排序
  -> 返回 top_k 个 Chunk
```

中文在这个基线版本中按单字匹配，英文和数字按连续单词匹配。这不是最终的中文分词方案，
但足以先验证数据格式、检索结果和评测流程。

## 运行

先确保已经生成 Chunk：

```powershell
python -m app.document_processor
python -m tests.retriever_test
```

也可以手动查看检索结果：

```powershell
python -m app.retriever
```

## 为什么先做关键词检索

关键词检索的优点是无 API 成本、速度快、结果容易解释。它可以作为后续向量检索的对照基线。
如果后面换成 Embedding 检索，仍然可以用同一批测试问题比较命中率和答案质量。
