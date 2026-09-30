# 对话记忆单测：滑动窗口 / 溢出缓冲 / 摘要压缩 / 前端历史合并
import json

from app.memory import store
from app.memory import summarizer


class FakePipeline:
    def __init__(self, fake):
        self.fake = fake
        self.ops = []

    def rpush(self, key, *values):
        self.ops.append(("rpush", key, values))

    def set(self, key, value, ex=None):
        self.ops.append(("set", key, value, ex))

    def delete(self, key):
        self.ops.append(("delete", key))

    def expire(self, key, ttl):
        self.ops.append(("expire", key, ttl))

    def execute(self):
        results = []
        for op in self.ops:
            name = op[0]
            if name == "rpush":
                results.append(self.fake.rpush(op[1], *op[2]))
            elif name == "set":
                results.append(self.fake.set(op[1], op[2], ex=op[3]))
            elif name == "delete":
                results.append(self.fake.delete(op[1]))
            elif name == "expire":
                results.append(True)
        self.ops = []
        return results


class FakeRedis:
    def __init__(self):
        self.data = {}

    def pipeline(self):
        return FakePipeline(self)

    def rpush(self, key, *values):
        self.data.setdefault(key, []).extend(values)
        return len(self.data[key])

    def llen(self, key):
        return len(self.data.get(key, []))

    def lpop(self, key):
        items = self.data.get(key)
        if not items:
            return None
        return items.pop(0)

    def lrange(self, key, start, end):
        items = self.data.get(key, [])
        if end < 0:
            return items[start:]
        return items[start:end + 1]

    def get(self, key):
        value = self.data.get(key)
        return value if isinstance(value, str) else None

    def set(self, key, value, ex=None):
        self.data[key] = value
        return True

    def delete(self, key):
        existed = key in self.data
        self.data.pop(key, None)
        return 1 if existed else 0

    def expire(self, key, ttl):
        return True

    def ping(self):
        return True


def _msg(role, content):
    return json.dumps({"role": role, "content": content}, ensure_ascii=False)


def _use_fake_redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(store, "get_redis", lambda: fake)
    return fake


def test_append_messages_sliding_window(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "memory_window", 3)
    fake = _use_fake_redis(monkeypatch)

    ok = store.append_messages("c1", [
        {"role": "user", "content": "m1"},
        {"role": "assistant", "content": "m2"},
        {"role": "user", "content": "m3"},
        {"role": "assistant", "content": "m4"},
        {"role": "user", "content": "m5"},
    ])
    assert ok is True

    context = store.load_context("c1")
    # 窗口 3：只保留最近 3 条
    assert [m["content"] for m in context["recent"]] == ["m3", "m4", "m5"]
    # 溢出的 2 条进入待摘要缓冲（时间正序）
    overflow = store.get_overflow("c1")
    assert [m["content"] for m in overflow] == ["m1", "m2"]


def test_append_without_redis_degrades(monkeypatch):
    monkeypatch.setattr(store, "get_redis", lambda: None)
    assert store.append_messages("c2", [{"role": "user", "content": "x"}]) is False
    assert store.load_context("c2") == {"summary": None, "recent": []}


def test_maybe_compress_folds_overflow_into_summary(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "memory_window", 2)
    monkeypatch.setattr(settings, "memory_summarize_threshold", 2)
    fake = _use_fake_redis(monkeypatch)

    store.append_messages("c3", [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
    ])
    assert len(store.get_overflow("c3")) >= 2

    summarizer_calls = []

    def fake_summarize(previous_summary, messages):
        summarizer_calls.append((previous_summary, [m["content"] for m in messages]))
        return "压缩后的摘要"

    monkeypatch.setattr(summarizer, "summarize_messages", fake_summarize)

    assert summarizer.maybe_compress("c3") is True

    context = store.load_context("c3")
    assert context["summary"] == "压缩后的摘要"
    assert store.get_overflow("c3") == []
    # 既有摘要为 None（首次压缩），溢出内容（u1/a1/u2）全部进入摘要；
    # a2 仍在滑窗原文保留，不应进入摘要
    previous, contents = summarizer_calls[0]
    assert previous is None
    assert "u1" in contents and "u2" in contents
    assert "a2" not in contents


def test_maybe_compress_llm_failure_keeps_overflow(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "memory_window", 2)
    monkeypatch.setattr(settings, "memory_summarize_threshold", 2)
    fake = _use_fake_redis(monkeypatch)

    store.append_messages("c4", [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ])
    monkeypatch.setattr(summarizer, "summarize_messages", lambda p, m: None)

    assert summarizer.maybe_compress("c4") is False
    # 失败时溢出缓冲保留，下次写回重试
    assert len(store.get_overflow("c4")) == 2


def test_merge_history_client_first(monkeypatch):
    client = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "r1"}]
    server = [
        {"role": "user", "content": "old1"},
        {"role": "assistant", "content": "r1"},  # 与前端重复
        {"role": "user", "content": "old2"},
    ]
    merged = store.merge_history(client, server)
    contents = [m["content"] for m in merged]
    # 前端历史优先且去重
    assert contents[0] == "q1" and contents[1] == "r1"
    assert contents.count("r1") == 1
    assert "old1" in contents and "old2" in contents


def test_merge_history_empty_client_uses_server(monkeypatch):
    server = [{"role": "user", "content": "old"}]
    assert store.merge_history([], server) == server
