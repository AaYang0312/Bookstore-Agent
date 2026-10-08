# 写操作确认门：propose 工具生成待确认操作，用户确认消息（卡片按钮或文字）消费并触发服务端执行
# LLM 只能"提议"写操作（propose_order / propose_cancel_order），真实执行只发生在
# 用户确认之后，由服务端用存储的原始参数直接调用执行器——机制级杜绝 LLM 自行下单
import contextvars
import json
import logging
import re
import time
import uuid

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

# 待确认操作保留时长（超时自动失效，用户点旧卡片会得到"已过期"提示）
PENDING_TTL_SECONDS = 300

KEY_PREFIX = "agent:pending"

# Redis 不可用时的单进程内存兜底（uvicorn 单进程部署下可用）
_memory_store: dict[str, tuple[str, float]] = {}

_conversation_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agent_conversation_id", default=None
)


def set_conversation_id(conversation_id: str | None) -> None:
    _conversation_var.set(conversation_id)


def get_conversation_id() -> str | None:
    return _conversation_var.get()


def _key(conversation_id: str) -> str:
    return f"{KEY_PREFIX}:{conversation_id}"


def create_pending(tool: str, arguments: dict, summary: dict) -> dict:
    """创建（覆盖）当前会话的待确认操作，返回完整操作对象。

    每个会话同时只有一个待确认操作：用户改主意重新 propose 时旧卡片自动作废。
    """
    conversation_id = get_conversation_id()
    if not conversation_id:
        raise ValueError("缺少 conversation_id，无法创建待确认操作")
    operation = {
        "op_id": uuid.uuid4().hex[:16],
        "tool": tool,
        "arguments": arguments,
        "summary": summary,
        "created_at": time.time(),
    }
    client = get_redis()
    if client is not None:
        client.set(_key(conversation_id), json.dumps(operation, ensure_ascii=False),
                   ex=PENDING_TTL_SECONDS)
    else:
        _gc_memory()
        _memory_store[conversation_id] = (
            json.dumps(operation, ensure_ascii=False), time.time() + PENDING_TTL_SECONDS)
    return operation


def get_pending(conversation_id: str) -> dict | None:
    client = get_redis()
    if client is not None:
        try:
            raw = client.get(_key(conversation_id))
        except Exception:
            return None
        return _decode(raw)
    entry = _memory_store.get(conversation_id)
    if not entry:
        return None
    raw, expires_at = entry
    if time.time() >= expires_at:
        _memory_store.pop(conversation_id, None)
        return None
    return _decode(raw)


def consume_pending(conversation_id: str, op_id: str | None = None) -> dict | None:
    """取出并删除待确认操作；op_id 为空时取最新（文字"确认"场景）。

    不匹配（过期/已被消费/op_id 不符）返回 None。
    """
    client = get_redis()
    if client is not None:
        try:
            raw = client.get(_key(conversation_id))
            operation = _decode(raw)
            if operation is None or not _op_matches(operation, op_id):
                return None
            client.delete(_key(conversation_id))
            return operation
        except Exception:
            return None
    entry = _memory_store.get(conversation_id)
    if not entry:
        return None
    raw, expires_at = entry
    if time.time() >= expires_at:
        _memory_store.pop(conversation_id, None)
        return None
    operation = _decode(raw)
    if operation is None or not _op_matches(operation, op_id):
        return None
    _memory_store.pop(conversation_id, None)
    return operation


def clear_pending(conversation_id: str) -> bool:
    """删除待确认操作（用户拒绝），返回是否确有操作被清除。"""
    client = get_redis()
    if client is not None:
        try:
            return bool(client.delete(_key(conversation_id)))
        except Exception:
            return False
    return _memory_store.pop(conversation_id, None) is not None


def _op_matches(operation: dict, op_id: str | None) -> bool:
    return op_id is None or operation.get("op_id") == op_id


def _decode(raw) -> dict | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        operation = json.loads(raw)
    except ValueError:
        return None
    if isinstance(operation, dict) and operation.get("op_id") and operation.get("tool"):
        return operation
    return None


def _gc_memory() -> None:
    now = time.time()
    for cid in [c for c, (_, exp) in _memory_store.items() if now >= exp]:
        _memory_store.pop(cid, None)


# ---- 确认消息解析 ----
# 卡片按钮发送精确标记（带 op_id）；用户手打的短语按"最新待确认操作"处理
_CONFIRM_MARKER = re.compile(r"^\[CONFIRM:([0-9a-f]{16})]$", re.IGNORECASE)
_REJECT_MARKER = re.compile(r"^\[REJECT:([0-9a-f]{16})]$", re.IGNORECASE)

CONFIRM_PHRASES = {"确认", "确定", "确认下单", "确认执行", "同意下单", "确认取消"}
REJECT_PHRASES = {"取消操作", "暂不下单", "先不买了", "先不下了"}


def parse_confirmation(message: str | None) -> tuple[str, str | None] | None:
    """识别确认/拒绝消息。

    Returns:
        ("confirm", op_id|None) / ("reject", op_id|None)；非确认消息返回 None。
        op_id 为 None 表示针对最新待确认操作（自然语言确认）。
    """
    text = (message or "").strip()
    if not text:
        return None
    match = _CONFIRM_MARKER.match(text)
    if match:
        return "confirm", match.group(1).lower()
    match = _REJECT_MARKER.match(text)
    if match:
        return "reject", match.group(1).lower()
    if text in CONFIRM_PHRASES:
        return "confirm", None
    if text in REJECT_PHRASES:
        return "reject", None
    return None
