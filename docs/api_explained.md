# HTTP API 入口

## 工具职责

- FastAPI：用 Python 定义 HTTP 路由、请求字段和响应字段。
- Uvicorn：启动 ASGI 服务进程，让其他程序可以通过 HTTP 调用 FastAPI。

## 安装依赖

在 `kbqa` 环境中执行：

```powershell
python -m pip install -r requirements-web.txt
```

## 启动服务

先生成最新 Chunk：

```powershell
python -m app.document_processor
python -m uvicorn app.api:app --reload
```

服务启动后：

- 健康检查：`http://127.0.0.1:8000/health`
- 交互文档：`http://127.0.0.1:8000/docs`
- 问答接口：`POST http://127.0.0.1:8000/chat`

请求示例：

```json
{
  "question": "年假多少天",
  "top_k": 3
}
```

API 会返回答案、分类、置信度和经过校验的引用列表。
