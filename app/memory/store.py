# Redis 会话读写：滑动窗口 + 溢出缓冲（供 summarizer 压缩），按 conversation_id 隔离
import json
import logging

from app.config import settings
from app.redis_client import get_redis

logger = logging.getLogger(__name__)

KEY_PREFIX = "chat:hist"


class MemoryStoreError(Exception):
    """会话存储异常（调用方应降级，不影响对话主流程）"""
    pass


def _keys(conversation_id: str) -> dict[str, str]:
    base = f"{KEY_PREFIX}:{conversation_id}"
    return {"recent": f"{base}:recent", "summary": f"{base}:summary", "overflow": f"{base}:overflow"}


def _refresh_ttl(client, keys: dict[str, str]) -> None:
    for key in keys.values():
        client.expire(key, settings.memory_ttl_seconds)


def append_messages(conversation_id: str, messages: list[dict]) -> bool:
    """追加消息到服务端会话；超窗部分移入溢出缓冲。失败返回 False（降级）。"""
    client = get_redis()
    if client is None or not messages:
        return False
    keys = _keys(conversation_id)
    try:
        pipe = client.pipeline()
        for message in messages:
            pipe.rpush(keys["recent"], json.dumps(message, ensure_ascii=False))
        pipe.execute()

        # 滑动窗口：最近 N 条保留原文，更早的转入溢出缓冲等待摘要
        overflowed = []
        while client.llen(keys["recent"]) > settings.memory_window:
            overflowed.append(client.lpop(keys["recent"]))
        if overflowed:
            client.rpush(keys["overflow"], *[m for m in overflowed if m])

        _refresh_ttl(client, keys)
        return True
    except Exception as e:
        logger.warning("写回会话记忆失败（降级为无记忆模式）：%s", e)
        return False


def load_context(conversation_id: str) -> dict:
    """读取服务端会话上下文：{"summary": str|None, "recent": [message]}。"""
    client = get_redis()
    empty = {"summary": None, "recent": []}
    if client is None:
        return empty
    keys = _keys(conversation_id)
    try:
        raw_recent = client.lrange(keys["recent"], 0, -1)
        summary = client.get(keys["summary"])
    except Exception as e:
        logger.warning("读取会话记忆失败（降级为无记忆模式）：%s", e)
        return empty

    recent = []
    for raw in raw_recent:
        try:
            message = json.loads(raw)
            if isinstance(message, dict) and message.get("role") and message.get("content"):
                recent.append({"role": message["role"], "content": message["content"]})
        except ValueError:
            continue
    return {"summary": summary if isinstance(summary, str) and summary else None, "recent": recent}


def get_overflow(conversation_id: str) -> list[dict]:
    """读取溢出缓冲（待摘要消息，时间正序）。"""
    client = get_redis()
    if client is None:
        return []
    keys = _keys(conversation_id)
    try:
        raw_list = client.lrange(keys["overflow"], 0, -1)
    except Exception:
        return []
    messages = []
    for raw in raw_list:
        try:
            message = json.loads(raw)
            if isinstance(message, dict) and message.get("role"):
                messages.append(message)
        except ValueError:
            continue
    return messages


def save_summary(conversation_id: str, summary: str) -> bool:
    """保存压缩摘要并清空溢出缓冲（摘要成功后调用，幂等）。"""
    client = get_redis()
    if client is None:
        return False
    keys = _keys(conversation_id)
    try:
        pipe = client.pipeline()
        pipe.set(keys["summary"], summary)
        pipe.delete(keys["overflow"])
        pipe.expire(keys["summary"], settings.memory_ttl_seconds)
        pipe.execute()
        return True
    except Exception as e:
        logger.warning("保存会话摘要失败：%s", e)
        return False


def merge_history(client_history: list[dict], server_recent: list[dict]) -> list[dict]:
    """前端传入 history 与服务端历史合并：前端历史优先入窗，服务端补充其未覆盖的消息。"""
    if not client_history:
        return server_recent

    seen = {(m.get("role"), m.get("content")) for m in client_history}
    merged = list(client_history)
    for message in server_recent:
        key = (message.get("role"), message.get("content"))
        if key not in seen:
            merged.append(message)
            seen.add(key)
    # 保留最近 memory_window 条
    return merged[-settings.memory_window:]
