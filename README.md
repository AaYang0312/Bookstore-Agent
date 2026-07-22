# 博学书城智能购书助手

基于 LLM 的图书店客服 Agent，通过 FastAPI 提供后端 API，支持调用工具查询书城真实图书数据。

## 功能特性

- 自然语言对话：以购书助手角色与用户交流
- 工具调用：通过 `search_books`、`get_book_detail` 工具查询书城真实库存、价格等信息
- 多轮工具调用：支持最多 5 轮工具调用，满足复杂查询需求
- OpenAI 兼容：使用 OpenAI Python SDK，兼容所有 OpenAI API 格式的服务

## 技术栈

- **FastAPI** — Web 框架
- **Pydantic** — 数据验证
- **OpenAI Python SDK** — 模型调用
- **pydantic_settings** — 配置管理
- **httpx** — HTTP 客户端（调用书城 Go 后端）

## 项目结构

```
app/
├── main.py              # FastAPI 路由入口
├── config.py            # 配置管理（pydantic_settings）
├── schemas.py           # Pydantic 数据模型
├── agent/
│   ├── harness.py       # LLM 调用与工具循环封装
│   └── prompt.py        # System Prompt 定义
└── tools/
    ├── registry.py      # 工具注册与执行调度
    └── book.py          # 图书搜索工具（调用 Go 后端 API）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

需要运行测试时安装开发依赖：

```bash
pip install -r requirements-dev.txt
pytest -q
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入实际配置：

```bash
cp .env.example .env
```

`.env` 配置项：

| 变量名 | 必填 | 说明 |
|--------|------|------|
| `MODEL_NAME` | 是 | 模型名称，如 `gpt-4o` |
| `MODEL_API_KEY` | 是 | 模型 API 密钥 |
| `MODEL_BASE_URL` | 否 | 模型 API 地址（用于非 OpenAI 官方服务） |
| `TAVILY_API_KEY` | 是 | Tavily 联网搜索 API 密钥 |
| `BOOKSTORE_API_BASE_URL` | 否 | 书城 Go 后端地址，默认 `http://localhost:8080/api/v1` |
| `BOOKSTORE_ADMIN_BASE_URL` | 否 | 书城管理 API 地址，默认 `http://localhost:8080/admin` |

### 3. 启动服务

```bash
uvicorn app.main:app --reload
```

服务启动后访问 `http://localhost:8000/docs` 查看 API 文档。

## Docker 快速部署

### 构建并单独运行 Agent

先按上文准备 `.env`，然后执行：

```bash
docker build -t bookstore-agent .
docker run --rm \
  --name bookstore-agent \
  -p 8000:8000 \
  --env-file .env \
  -e BOOKSTORE_API_BASE_URL=http://host.docker.internal:8080/api/v1 \
  bookstore-agent
```

以上命令适用于 Docker Desktop，且 Go 后端运行在宿主机 8080 端口。Linux 环境请改用容器网络中的后端服务名，或显式配置宿主机网关。镜像以非 root 用户运行，并通过 `GET /health` 进行健康检查。

### 一键启动三个项目

推荐使用前端仓库中的 Compose 编排，它会同时启动前端、Go 后端、Agent、MySQL 和 Redis，并自动把 `BOOKSTORE_API_BASE_URL` 设置为容器内后端地址：

```bash
cd ../bookstore-fronted-master
cp .env.docker.example .env.docker
# 编辑 .env.docker，填写 MODEL_* 和 TAVILY_API_KEY
docker compose --env-file .env.docker up --build -d
docker compose ps
```

启动后可访问 `http://localhost:8000/health` 和 `http://localhost:8000/docs`。占位密钥可以让服务通过健康检查，但实际对话和联网搜索必须使用有效密钥。

## API 接口

### 对话接口

```
POST /api/v1/agent/chat
```

**请求体：**

```json
{
  "message": "有没有《三体》？",
  "conversation_id": "conv_001",
  "history": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你的？"}
  ]
}
```

**响应：**

```json
{
  "message": "找到了《三体》这本书...",
  "conversation_id": "conv_001"
}
```

### 健康检查

```
GET /health
```

### 模型信息

```
GET /model-info
```

## 工具说明

当前已实现的工具：

| 工具名 | 功能 | 参数 |
|--------|------|------|
| `search_books` | 按关键词搜索图书 | `keyword`（必填）、`page`、`page_size` |
| `get_book_detail` | 查看图书详情 | `book_id`（必填，书籍 ID，从 1 开始） |

工具调用流程：模型决定调用工具 → Agent 执行工具并获取结果 → 将结果返回模型 → 模型生成最终回答。
