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


@app.post("/api/v1/agent/chat", response_model=AgentChatResponse)
def chat(request: AgentChatRequest):
    history, summary = _assemble_context(request.conversation_id,
                                         [m.model_dump() for m in request.history])
    profile_segment = None
    if get_token():
        profile_segment = build_profile_segment()

    try:
        reply = harness.model_call(
            history=history,
            message=request.message,
            profile_segment=profile_segment,
            summary=summary,
        )
    except RuntimeError as e:
        reply = f"抱歉，处理您的请求时出现问题：{e}"

    _write_back(request.conversation_id, request.message, reply)

    return AgentChatResponse(
        message=reply,
        conversation_id=request.conversation_id,
    )


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


TOOL_RESULT_SUMMARY_MAX_CHARS = 300


@app.get("/api/v1/agent/chat/stream")
def chat_stream(
    message: str = Query(min_length=1, max_length=1500),
    conversation_id: str | None = Query(default=None, max_length=100),
):
    """SSE 流式对话：event 序列为 delta / tool_start / tool_end / done / error。

    支持 Authorization: Bearer <JWT> 头（个性化与用户数据工具）。
    conversation_id 缺省时由服务端生成并在 done 事件返回。
    """
    if not conversation_id:
        conversation_id = uuid.uuid4().hex
    token = get_token()

    def event_generator():
        history, summary = _assemble_context(conversation_id, [])
        profile_segment = build_profile_segment(token) if token else None

        final_text = None
        try:
            for event in harness.run_agent(
                history, message, profile_segment=profile_segment, summary=summary,
            ):
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
                elif event["type"] == "final":
                    final_text = event["content"]
                    # 写回与 done 解耦：断连前完成记忆写回（幂等：user+assistant 一次写入）
                    _write_back(conversation_id, message, final_text)
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
