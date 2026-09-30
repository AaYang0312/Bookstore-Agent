# SSE 流式 + 同步接口回归：事件序列 / 断连安全 / 同步路径兼容
import json

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app


def _parse_sse(text: str):
    """把 SSE 文本解析为 (event, data) 列表。"""
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.strip().split("\n")
        event, data = None, None
        for line in lines:
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event:
            events.append((event, data))
    return events


def _fake_events():
    yield {"type": "delta", "content": "你好"}
    yield {"type": "delta", "content": "！"}
    yield {"type": "tool_start", "name": "search_books", "arguments": '{"keyword": "三体"}'}
    yield {"type": "tool_end", "name": "search_books", "result": '{"ok": true}'}
    yield {"type": "final", "content": "你好！为您找到三体"}


def test_sse_stream_event_order(monkeypatch):
    monkeypatch.setattr(main_module.harness, "run_agent",
                        lambda history, message, profile_segment=None, summary=None: _fake_events())
    client = TestClient(app)

    response = client.get("/api/v1/agent/chat/stream",
                          params={"message": "帮我找三体", "conversation_id": "sse-test-1"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    sequence = [event for event, _ in events]

    # 事件序列完整有序
    assert sequence == ["delta", "delta", "tool_start", "tool_end", "done"]
    assert events[0][1] == {"content": "你好"}
    assert events[2][1] == {"name": "search_books", "arguments": {"keyword": "三体"}}
    assert events[4][1]["message"] == "你好！为您找到三体"
    assert events[4][1]["conversation_id"] == "sse-test-1"


def test_sse_stream_generates_conversation_id(monkeypatch):
    monkeypatch.setattr(main_module.harness, "run_agent",
                        lambda history, message, profile_segment=None, summary=None: _fake_events())
    client = TestClient(app)

    response = client.get("/api/v1/agent/chat/stream", params={"message": "hi"})

    events = _parse_sse(response.text)
    done = [data for event, data in events if event == "done"][0]
    assert done["conversation_id"]  # 服务端生成


def test_sse_error_event(monkeypatch):
    def failing(history, message, profile_segment=None, summary=None):
        yield {"type": "delta", "content": "开头"}
        yield {"type": "error", "message": "模型返回空响应"}

    monkeypatch.setattr(main_module.harness, "run_agent", failing)
    client = TestClient(app)

    response = client.get("/api/v1/agent/chat/stream", params={"message": "hi"})

    events = _parse_sse(response.text)
    assert [event for event, _ in events] == ["delta", "error"]
    assert "空响应" in events[-1][1]["message"]


def test_sse_message_required():
    client = TestClient(app)
    assert client.get("/api/v1/agent/chat/stream").status_code == 422


def test_sync_chat_keeps_contract(monkeypatch):
    monkeypatch.setattr(main_module.harness, "model_call",
                        lambda history, message, profile_segment=None, summary=None: "同步回复")
    client = TestClient(app)

    response = client.post("/api/v1/agent/chat", json={
        "message": "推荐一本书",
        "conversation_id": "sync-test-1",
        "history": [{"role": "user", "content": "之前问过科幻"}],
    })

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "同步回复"
    assert body["conversation_id"] == "sync-test-1"


def test_sync_chat_error_degrades_to_friendly_message(monkeypatch):
    def raising(history, message, profile_segment=None, summary=None):
        raise RuntimeError("工具调用次数过多")

    monkeypatch.setattr(main_module.harness, "model_call", raising)
    client = TestClient(app)

    response = client.post("/api/v1/agent/chat", json={
        "message": "hi", "conversation_id": "sync-test-2",
    })

    assert response.status_code == 200
    assert "抱歉" in response.json()["message"]


def test_health_reports_components():
    client = TestClient(app)
    body = client.get("/health").json()
    assert body["status"] in ("ok", "degraded")
    assert "redis" in body["components"]
    assert "milvus" in body["components"]
    assert "rag" in body["components"]
