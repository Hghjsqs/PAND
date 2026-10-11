# 数据说明

## `data/raw/`

第一批是虚构的企业制度文档，所有文档均标注了文档编号、版本、生效日期和负责部门。

当前文件：

- `leave_policy.md`：年假额度、申请流程和审批；
- `reimbursement_policy.md`：报销材料、时限和金额审批；
- `it_support.md`：网络、账号和 IT 工单处理；
- `purchase_policy.md`：采购申请、金额审批和供应商比价。

## `docs/evaluation_questions.jsonl`

每行是一个 JSON 对象，字段如下：

| 字段 | 说明 |
|---|---|
| `id` | 测试问题唯一 ID |
| `question` | 用户问题 |
| `answer` | 预期答案或拒答行为 |
| `source` | 标准证据文件；无答案问题为 `null` |
| `section` | 标准证据章节；无答案问题为 `null` |
| `type` | 问题类型 |

当前测试集共 14 题：12 个有依据问题、1 个文档无答案问题、1 个范围外问题。

测试集不是让模型背答案，而是用来比较不同切块、Embedding、BM25、Hybrid 和 Rerank 方案的效果。

后续新增问题时不要删除旧问题，否则不同版本的指标无法比较。若制度版本发生变化，应新增测试问题或记录对应版本。
