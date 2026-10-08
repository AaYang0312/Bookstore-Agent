# FAST API 路由：同步 /chat（兼容）+ SSE /chat/stream + 健康检查
import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.agent import harness
from app.auth_context import get_token, set_token
from app.config import settings
from app.memory import store as memory_store
from app.memory.summarizer import maybe_compress
from app.personalization.profile import build_profile_segment
from app.schemas import AgentChatRequest, AgentChatResponse
from app.tools import write_gate

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = None
    if settings.rag_ready and settings.ingest_schedule_enabled:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler

            from app.rag.ingest import scheduled_incremental

            scheduler = BackgroundScheduler(daemon=True)
            scheduler.add_job(
                scheduled_incremental,
                "interval",
                minutes=settings.ingest_schedule_interval_minutes,
                id="rag_incremental_ingest",
                max_instances=1,
                coalesce=True,
            )
            scheduler.start()
            logger.info("RAG 定时增量入库已启用（每 %d 分钟）", settings.ingest_schedule_interval_minutes)
        except Exception as e:
            logger.warning("启动定时入库失败：%s", e)
    yield
    if scheduler is not None:
        scheduler.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def auth_context_middleware(request: Request, call_next):
    """提取 Authorization 头到请求级上下文，供用户数据工具透传给 Go 后端。"""
    authorization = request.headers.get("Authorization")
    if authorization and authorization.lower().startswith("bearer "):
        set_token(authorization[7:].strip())
    else:
        set_token(None)
    return await call_next(request)


def _component_status() -> dict:
    """组件连通状态：失败只降级标记，不影响 200。"""
    components = {}
    try:
        from app import redis_client
        from app.rag import vector_store

        redis_ok = redis_client.ready()
        components["redis"] = {"ok": redis_ok, "configured": bool(settings.redis_url)}
        milvus_configured = bool(settings.milvus_uri)
        milvus_ok = vector_store.ready() if milvus_configured else False
        components["milvus"] = {"ok": milvus_ok, "configured": milvus_configured}
        components["rag"] = {
            "enabled": settings.rag_ready,
            "semantic_search_books": settings.rag_ready,
        }
    except Exception as e:
        logger.warning("健康检查组件探测异常：%s", e)
    return components


@app.get("/health")
def health_check():
    components = _component_status()
    degraded = [name for name, status in components.items()
                if status.get("configured") and not status.get("ok")]
    return {
        "status": "ok" if not degraded else "degraded",
        "components": components,
    }


def _assemble_context(conversation_id: str, client_history: list[dict]) -> tuple[list[dict], str | None]:
    """组装对话上下文：服务端记忆优先，前端传入 history 合并入窗（兼容降级）。"""
    context = memory_store.load_context(conversation_id)
    server_recent = context.get("recent", [])
    history = memory_store.merge_history(client_history, server_recent)
    summary = context.get("summary")
    return history, summary


def _write_back(conversation_id: str, user_message: str, assistant_message: str) -> None:
    """统一在完成后写回记忆（user+assistant 一次写入，断连不产生半写会话）。"""
    ok = memory_store.append_messages(conversation_id, [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ])
    if ok:
        maybe_compress(conversation_id)


# 确认卡片写回时使用的友好用户文案（替代原始 [CONFIRM:...] 标记）
_CONFIRMED_USER_TEXT = {
    "create_order": "（确认创建订单）",
    "cancel_order": "（确认取消订单）",
}


def _confirmation_branch(conversation_id: str, message: str) -> tuple[str, dict | None]:
    """识别确认/拒绝消息，返回 (分支, 待执行操作)。

    分支：normal（普通对话）/ confirmed（用户确认，执行存储的操作）/
    rejected（用户拒绝，清除待确认操作）。无匹配操作时回落 normal。
    """
    intent = write_gate.parse_confirmation(message)
    if intent is None:
        return "normal", None
    kind, op_id = intent
    if kind == "confirm":
        operation = write_gate.consume_pending(conversation_id, op_id)
        if operation is not None:
            return "confirmed", operation
        return "normal", None
    if write_gate.clear_pending(conversation_id):
        return "rejected", None
    return "normal", None


@app.post("/api/v1/agent/chat", response_model=AgentChatResponse)
def chat(request: AgentChatRequest):
    history, summary = _assemble_context(request.conversation_id,
                                         [m.model_dump() for m in request.history])
    write_gate.set_conversation_id(request.conversation_id)
    profile_segment = None
    if get_token():
        profile_segment = build_profile_segment()

    branch, operation = _confirmation_branch(request.conversation_id, request.message)

    try:
        if branch == "confirmed":
            # 用户已确认：服务端直接执行存储的写操作，LLM 基于真实结果作答
            reply = harness.model_call_confirmed(
                history=history,
                operation=operation,
                profile_segment=profile_segment,
                summary=summary,
            )
            user_text = _CONFIRMED_USER_TEXT.get(operation["tool"], "（确认执行操作）")
        else:
            message = request.message
            if branch == "rejected":
                message = "（用户取消了刚才提议的操作，请友好回应，不要执行任何写操作）"
            reply = harness.model_call(
                history=history,
                message=message,
                profile_segment=profile_segment,
                summary=summary,
            )
            user_text = request.message if branch == "normal" else "（取消操作）"
    except RuntimeError as e:
        reply = f"抱歉，处理您的请求时出现问题：{e}"
        user_text = request.message

    _write_back(request.conversation_id, user_text, reply)

    return AgentChatResponse(
        message=reply,
        conversation_id=request.conversation_id,
    )


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


TOOL_RESULT_SUMMARY_MAX_CHARS = 300


@app.get("/api/v1/agent/chat/stream")
async def chat_stream(
    message: str = Query(min_length=1, max_length=1500),
    conversation_id: str | None = Query(default=None, max_length=100),
):
    """SSE 流式对话：event 序列为 delta / tool_start / tool_end / confirm_request / done / error。

    支持 Authorization: Bearer <JWT> 头（个性化与用户数据工具）。
    conversation_id 缺省时由服务端生成并在 done 事件返回。
    """
    conversation_id = conversation_id or uuid.uuid4().hex
    # 必须在 async 端点（ASGI 任务上下文）内设置：SSE 生成器经线程池逐段迭代，
    # 每段从任务上下文重新拷贝，生成器内部 set 的值无法跨迭代存活
    write_gate.set_conversation_id(conversation_id)
    return _stream_response(message, conversation_id, [])


@app.post("/api/v1/agent/chat/stream")
async def chat_stream_post(request: AgentChatRequest):
    """POST 版 SSE 流式对话（请求体与同步 /chat 一致，可携带 history）。

    供前端统一入口使用：响应仍为 text/event-stream。
    """
    conversation_id = request.conversation_id or uuid.uuid4().hex
    write_gate.set_conversation_id(conversation_id)
    return _stream_response(
        request.message,
        conversation_id,
        [m.model_dump() for m in request.history],
    )


def _stream_response(message: str, conversation_id: str, client_history: list[dict]):
    token = get_token()

    def event_generator():
        history, summary = _assemble_context(conversation_id, client_history)
        profile_segment = build_profile_segment(token) if token else None

        final_text = None
        try:
            branch, operation = _confirmation_branch(conversation_id, message)
            if branch == "confirmed":
                events = harness.run_confirmed(
                    history, operation, profile_segment=profile_segment, summary=summary,
                )
                user_text = _CONFIRMED_USER_TEXT.get(operation["tool"], "（确认执行操作）")
            else:
                agent_message = message
                if branch == "rejected":
                    agent_message = "（用户取消了刚才提议的操作，请友好回应，不要执行任何写操作）"
                events = harness.run_agent(
                    history, agent_message,
                    profile_segment=profile_segment, summary=summary,
                )
                user_text = message if branch == "normal" else "（取消操作）"

            for event in events:
                if event["type"] == "delta":
                    yield _sse_event("delta", {"content": event["content"]})
                elif event["type"] == "tool_start":
                    try:
                        arguments = json.loads(event["arguments"]) if event["arguments"] else {}
                    except json.JSONDecodeError:
                        arguments = {"raw": event["arguments"]}
                    yield _sse_event("tool_start", {"name": event["name"], "arguments": arguments})
                elif event["type"] == "tool_end":
                    result = event["result"] or ""
                    yield _sse_event("tool_end", {
                        "name": event["name"],
                        "result": result[:TOOL_RESULT_SUMMARY_MAX_CHARS],
                    })
                elif event["type"] == "confirm_request":
                    # 写操作确认卡片：前端渲染确认/取消按钮，按钮回发确认标记消息
                    yield _sse_event("confirm_request", {
                        "operation_id": event["operation_id"],
                        "summary": event["summary"],
                    })
                elif event["type"] == "final":
                    final_text = event["content"]
                    # 写回与 done 解耦：断连前完成记忆写回（幂等：user+assistant 一次写入）
                    _write_back(conversation_id, user_text, final_text)
                    yield _sse_event("done", {
                        "message": final_text,
                        "conversation_id": conversation_id,
                    })
                elif event["type"] == "error":
                    yield _sse_event("error", {"message": event["message"]})
                    return
        except Exception as e:
            logger.exception("SSE 流式对话异常")
            yield _sse_event("error", {"message": f"流式对话异常：{e}"})
            return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/model-info")
def get_model_info():
    return {
        "model": settings.model_name,
    }

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization"]
)
