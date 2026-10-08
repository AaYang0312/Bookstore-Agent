# 博学书城智能购书助手

基于 LLM 的图书店客服 Agent，通过 FastAPI 提供后端 API。在手写 OpenAI 工具循环之上扩展了
**RAG 语义检索**、**个性化推荐**、**服务端对话记忆** 与 **SSE 流式输出** 四大能力。

## 功能特性

- 自然语言对话：以购书助手角色与用户交流
- 工具调用：14 个工具覆盖关键词搜索、语义找书、图书详情、联网搜索、用户画像/订单/收藏/浏览，以及收藏管理、代下单、取消订单等写操作（带确认策略）
- 写操作闭环：可代用户创建**待支付订单**（不代支付，30 分钟未支付自动取消）与取消订单；下单走 Kafka 异步受理，工具内部轮询结果；幂等键防重复下单；Prompt 强制"复述书目与总价并获用户确认后才可下单"
- RAG 知识库：Milvus 向量检索 + 可选 Rerank 重排，支持"模糊找书"语义命中
- 个性化推荐：JWT 透传拉取书城聚合画像，注入 System Prompt（无登录自动降级）
- 对话记忆：Redis 滑动窗口 + LLM 摘要两级记忆，跨请求上下文连贯，前端传 history 仍兼容
- 流式输出：SSE 逐 token 推送，工具调用过程可见（`delta` / `tool_start` / `tool_end` / `done` / `error`）
- 增量同步：基于 `updated_at` 水位的向量库增量入库管道（CLI + 可选定时任务）
- 优雅降级：未配置 Milvus/Embedding/Redis 时相关工具与能力自动降级，原有三工具行为不变

## 架构

```mermaid
flowchart LR
    subgraph Client["客户端（本次改造范围外）"]
        FE[React 前端 / curl / Swagger]
    end

    subgraph Agent["Bookstore-Agent (FastAPI)"]
        API[main.py<br/>/chat · /chat/stream SSE]
        CTX[auth_context<br/>JWT 请求级上下文]
        MEM[memory<br/>滑窗 + 摘要两级记忆]
        PERS[personalization<br/>画像拉取 + TTL 缓存]
        HAR[harness<br/>流式工具循环生成器]
        TOOLS[tools × 14<br/>检索/联网/用户数据/收藏/订单(含写操作)]
        RAG[rag<br/>embedder · vector_store · retriever · ingest]
    end

    subgraph Go["Bookstore_Backend (Go/Gin)"]
        PUB[公开接口 /book/*]
        USER[JWT 接口 /user/* /order/* /favorite/*]
        BROWSE[browse_logs 埋点<br/>详情页自动 + 显式接口]
        SYNC[/internal/books/sync<br/>共享密钥只读同步]
    end

    subgraph Storage
        MYSQL[(MySQL)]
        REDIS[(Redis<br/>会话/画像缓存/水位/去重)]
        MILVUS[(Milvus<br/>HNSW · IP)]
    end

    EMB[Embedding API<br/>OpenAI 兼容]
    LLM[LLM API<br/>OpenAI 兼容]
    TAVILY[Tavily 搜索]

    FE -->|chat / SSE / JWT| API
    API --> CTX --> TOOLS
    API <--> MEM <--> REDIS
    API --> PERS --> USER
    HAR <--> LLM
    HAR --> TOOLS
    TOOLS --> PUB
    TOOLS --> USER
    PERS --> USER
    RAG <--> MILVUS
    RAG --> EMB
    RAG --> SYNC
    SYNC --> MYSQL
    BROWSE --> MYSQL
    BROWSE --> REDIS
    TOOLS --> TAVILY
```

## 技术栈

- **FastAPI** — Web 框架（SSE 通过 StreamingResponse）
- **OpenAI Python SDK** — 模型调用 / 兼容 Embedding 接口
- **pymilvus** — Milvus 向量库（HNSW 索引，IP 度量 + 归一化向量，标量混合过滤）
- **redis-py** — 会话记忆 / 画像缓存 / 入库水位
- **APScheduler** — 可选的定时增量入库
- **httpx** — HTTP 客户端（调用书城 Go 后端）
- **pydantic / pydantic_settings** — 数据验证与配置管理

## 项目结构

```
app/
├── main.py                     # 路由：/chat(同步) /chat/stream(SSE) /health /model-info
├── config.py                   # 配置（Milvus/Embedding/Redis/Rerank/记忆/画像）
├── schemas.py                  # 请求/响应模型 + SSE 事件模型
├── auth_context.py             # 请求级 JWT 上下文（ContextVar）
├── redis_client.py             # Redis 惰性单例（未配置自动降级）
├── agent/
│   ├── harness.py              # 流式生成器工具循环（同步/SSE 共用核心）
│   ├── prompt.py               # System Prompt
│   └── tool_prompt.py          # 工具策略 + 个性化回答策略
├── rag/
│   ├── embedder.py             # Embedding 客户端（批量/重试/维度自检/L2 归一化）
│   ├── vector_store.py         # Milvus 封装（建删/upsert/混合检索）
│   ├── ingest.py               # 入库管道 CLI（--full / --incremental）
│   └── retriever.py            # 召回 → 可选 Rerank → Top-K
├── memory/
│   ├── store.py                # Redis 会话（滑窗 + 溢出缓冲）
│   └── summarizer.py           # 超窗历史 LLM 摘要（保留偏好/已荐书/未决问题）
├── personalization/
│   └── profile.py              # 画像拉取（60s 缓存）→ Prompt 画像段
└── tools/
    ├── registry.py             # 工具注册（RAG 未配置时 semantic 工具不注册）
    ├── bookstore_api.py        # 书城 HTTP 共享层（JWT 透传 + 非 2xx 业务 message 保留）
    ├── book.py / web.py        # 原有三工具（行为不变）
    ├── rag_search.py           # semantic_search_books（语义找书）
    ├── user.py                 # 画像/订单/收藏/浏览四工具（无 JWT 结构化降级）
    ├── favorite.py             # 收藏新增/移除/查询（异步受理，check 可复核）
    └── order.py                # 代下单(轮询终态)/取消订单/订单详情
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 开发/测试
pip install -r requirements-dev.txt
pytest -q
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`。基础配置（原有能力）：

| 变量名 | 必填 | 说明 |
|--------|------|------|
| `MODEL_NAME` | 是 | 模型名称，如 `deepseek-chat` |
| `MODEL_API_KEY` | 是 | 模型 API 密钥 |
| `MODEL_BASE_URL` | 否 | 模型 API 地址（OpenAI 兼容服务） |
| `TAVILY_API_KEY` | 是 | Tavily 联网搜索 API 密钥 |
| `BOOKSTORE_API_BASE_URL` | 否 | 书城 Go 后端地址，默认 `http://localhost:8080/api/v1` |

扩展配置（全部可选，不配置则自动降级）：

| 变量名 | 说明 |
|--------|------|
| `MILVUS_URI` | Milvus 地址（如 `http://localhost:19530`），配置后启用语义检索 |
| `EMBEDDING_MODEL_NAME` | OpenAI 兼容 embedding 模型名，与 MILVUS_URI 同时配置才启用 RAG |
| `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` | 默认复用 `MODEL_*` |
| `EMBEDDING_DIM` | 向量维度（默认 1024）；collection 名含维度，切换模型即换集合并全量重灌 |
| `RAG_RECALL_K` / `RAG_TOP_K` | 召回/最终结果数（默认 20/5） |
| `RERANK_ENABLED` 等 | Rerank 重排（默认关闭，开启失败自动回退向量排序） |
| `REDIS_URL` | Redis 地址；启用服务端记忆/画像缓存/入库水位 |
| `INGEST_SCHEDULE_ENABLED` | 定时增量入库开关（默认关） |
| `INTERNAL_SYNC_SECRET` | Go 内部同步接口共享密钥（与后端 `BOOKSTORE_INTERNAL_SYNC_SECRET` 一致） |

### 3. 启动基础设施与全栈

前端仓库的 Compose 已编排全栈（含 Milvus standalone 及其 etcd/专用 MinIO）：

```bash
cd ../Bookstore-Front
docker compose up -d          # mysql redis kafka minio etcd milvus-minio milvus backend agent frontend
docker compose ps             # 全部 healthy
```

### 4. RAG 入库

```bash
# 全量入库（万级图书分钟级完成）
python -m app.rag.ingest --full

# 按 updated_at 水位增量（水位存 Redis）
python -m app.rag.ingest --incremental

# 首次无水位时指定起点
python -m app.rag.ingest --incremental --since "2026-09-01T00:00:00+08:00"
```

数据源走 Go 后端内部只读接口 `GET /internal/books/sync`（共享密钥鉴权），
不占用带 IP 限流的 `/book/search`。

### 5. 启动服务

```bash
uvicorn app.main:app --reload
```

访问 `http://localhost:8000/docs` 查看交互式 API 文档。

## API 接口

### 同步对话（向后兼容）

```
POST /api/v1/agent/chat
Authorization: Bearer <JWT>   （可选；提供后启用个性化）
```

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

响应（与旧版一致）：

```json
{ "message": "找到了《三体》……", "conversation_id": "conv_001" }
```

### SSE 流式对话

```
GET /api/v1/agent/chat/stream?message=想找讲宇宙文明的科幻小说&conversation_id=conv_001
Authorization: Bearer <JWT>   （可选）
```

curl 示例：`curl -N "http://localhost:8000/api/v1/agent/chat/stream?message=推荐几本书"`

事件序列：

```
event: delta
data: {"content": "为您找到"}

event: tool_start
data: {"name": "semantic_search_books", "arguments": {"query": "宇宙文明 科幻"}}

event: tool_end
data: {"name": "semantic_search_books", "result": "{\"ok\": true, ...}"}

event: done
data: {"message": "完整回答……", "conversation_id": "conv_001"}
```

错误以 `event: error` 推送；`conversation_id` 缺省时由服务端生成并在 `done` 中返回。
记忆写回在 done 时统一执行（user+assistant 一次写入），中断连接不产生半写会话。

### 健康检查

```
GET /health
```

返回各组件状态（redis/milvus/rag），组件故障只降级标记不影响 200。

## 工具说明

| 工具名 | 功能 | 说明 |
|--------|------|------|
| `search_books` | 关键词搜索图书 | 精确书名/作者/关键词 |
| `get_book_detail` | 图书详情 | 需先从检索结果获得真实 book_id |
| `search_web` | Tavily 联网搜索 | 新书资讯/书评 |
| `semantic_search_books` | 语义找书（RAG） | 模糊自然语言需求；支持 `category`/`max_price` 过滤；未配置 Milvus/Embedding 时不注册 |
| `get_user_profile` | 用户资料 | 需 JWT；未登录返回结构化降级提示 |
| `get_user_orders` | 用户订单 | 需 JWT |
| `get_user_favorites` | 用户收藏 | 需 JWT |
| `get_browse_history` | 浏览记录 | 需 JWT（数据来自 Go 后端浏览埋点） |
| `add_favorite` | 收藏图书（写） | 需 JWT；异步受理；book_id 须来自检索结果 |
| `remove_favorite` | 取消收藏（写） | 需 JWT；异步受理 |
| `check_favorite` | 查询是否已收藏 | 需 JWT；用于复核异步收藏结果 |
| `create_order` | 代创建待支付订单（写） | 需 JWT；Prompt 强制先复述书目/数量/总价并获确认；工具生成幂等键，LLM 重试不重复下单；内部轮询 Kafka 异步结果；不代支付 |
| `cancel_order` | 取消待支付订单（写） | 需 JWT；仅待支付可取消（幂等）；须先向用户确认订单 |
| `get_order_detail` | 单笔订单详情 | 需 JWT；含订单内书目/金额/状态 |

工具描述明确分工"语义模糊找书用 semantic_search_books，精确书名用 search_books"，
避免 LLM 路由选错工具；写操作工具（下单/取消/收藏）的描述内嵌确认前提，
配合 Tool Prompt 的"写操作与确认策略"（下单前必须复述书目与总价并获得用户明确同意）。

## RAG 效果验证

固定 20 条测试问题（同义改写/模糊描述），对比关键词搜索与语义检索命中率：

- 例："想找讲宇宙文明的科幻小说" → 语义检索命中《三体》，关键词搜索无结果
- 统计脚本可基于 `semantic_search_books` 与 `search_books` 的返回结果自动化生成对比报告

## 简历亮点

- 基于 **Milvus** 构建图书语义检索知识库，设计 HNSW 索引（IP 度量 + 向量归一化）与
  分类/价格标量混合过滤，支撑万级图书毫秒级语义召回
- 设计可切换的 **Embedding 服务层**（维度自检/批量/指数退避重试），实现
  "文档组装 → 向量化 → upsert" 的全量/增量入库管道（updated_at 水位 + 定时任务）
- 落地 **RAG** 方案：向量召回 Top20 → 可选 Rerank 重排 → Top-K 上下文注入，
  以固定测试问题集量化关键词 vs 语义命中率
- 扩展 Agent 工具集至 **14 个**（语义检索/资料/订单/收藏/浏览 + 收藏管理/代下单/取消订单），
  实现"意图确认 → 幂等提交 → 异步结果轮询"的 **Agent 写操作闭环**（LLM 重试不重复下单），
  完善参数校验、错误降级与多轮工具编排；同步与 SSE 流式共用单一生成器核心，避免两套逻辑漂移
- 打通 **JWT 透传**链路，聚合订单/收藏/浏览行为构建用户画像注入 Prompt，
  实现个性化荐书与匿名降级策略
- 设计 **滑动窗口 + LLM 摘要**的两级对话记忆（摘要保留用户偏好/已荐书/未决问题），
  长会话 Token 消耗受控且跨请求上下文连贯
- 实现 **SSE 流式响应协议**（delta/tool_start/tool_end/done 事件），
  首字延迟从整段生成降至首 token；断连不产生半写会话
- 在 **Go/Gin** 微服务中新增行为埋点（Redis SETNX 去重防刷、唯一索引 upsert）
  与聚合画像接口，为推荐提供数据基建；Docker Compose 编排
  Milvus/MySQL/Redis/Go/Agent 多服务一键部署

## 范围外（后续演进）

- 前端改造（SSE UI、JWT 附带、显式埋点调用）—— 接口已预留
- LangGraph / 多 Agent 协作
- Embedding 本地推理（BGE-M3 本地化）、Milvus 分布式集群
