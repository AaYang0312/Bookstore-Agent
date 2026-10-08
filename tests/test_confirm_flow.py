# 确认卡片机制端到端单测：propose 生成卡片 / LLM 不可见真实执行器 /
# run_confirmed 服务端执行 / SSE confirm_request 事件 / 同步确认分支
import json

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.tools import order as order_module, write_gate
from app.tools.order import OrderAPIError, propose_cancel_order, propose_order
from app.tools.registry import TOOLS

from test_sse_stream import _parse_sse


def _detail(title="三体", price=4400, discount=0, stock=10, book_id=9):
    current = price * (100 - discount) // 100
    return {"ok": True, "book": {
        "id": book_id, "title": title, "author": "刘慈欣", "price": price,
        "discount": discount, "current_price": current, "stock": stock,
    }}


def _setup(monkeypatch, book_detail=None, token="jwt-1", conversation="confirm-flow"):
    from app.tools import bookstore_api

    monkeypatch.setattr(write_gate, "get_redis", lambda: None)
    write_gate._memory_store.clear()
    write_gate.set_conversation_id(conversation)
    monkeypatch.setattr(order_module, "get_token", lambda: token)
    monkeypatch.setattr(bookstore_api, "get_token", lambda: token)
    if book_detail is not None:
        monkeypatch.setattr(order_module, "get_book_detail", lambda book_id: book_detail)


# ---- propose_order / propose_cancel_order ----

def test_propose_order_builds_card_and_pending(monkeypatch):
    _setup(monkeypatch, book_detail=_detail(price=4400, discount=50, stock=10))

    result = propose_order(items=[{"book_id": 9, "quantity": 2}])

    assert result["ok"] is True
    assert result["action"] == "confirm_required"
    summary = result["summary"]
    assert summary["items"][0]["unit_price"] == 2200   # 4400 * (100-50)/100
    assert summary["items"][0]["subtotal"] == 4400
    assert summary["estimated_total"] == 4400
    # 待确认操作已存储，携带原始参数（确认后服务端原样执行）
    stored = write_gate.get_pending("confirm-flow")
    assert stored["tool"] == "create_order"
    assert stored["arguments"] == {"items": [{"book_id": 9, "quantity": 2}]}


def test_propose_order_rejects_insufficient_stock(monkeypatch):
    _setup(monkeypatch, book_detail=_detail(stock=1))

    with pytest.raises(OrderAPIError, match="库存不足"):
        propose_order(items=[{"book_id": 9, "quantity": 2}])


def test_propose_order_requires_jwt(monkeypatch):
    _setup(monkeypatch, book_detail=_detail(), token=None)

    with pytest.raises(OrderAPIError, match="用户未登录"):
        propose_order(items=[{"book_id": 9, "quantity": 1}])


def test_propose_cancel_order_rejects_paid_order(monkeypatch):
    import httpx
    from app.tools import bookstore_api

    _setup(monkeypatch, conversation="cancel-flow")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 0, "data": {
            "id": 5, "order_no": "ORD5", "total_amount": 100, "status": 1, "is_paid": True,
            "order_items": [],
        }})

    real_client = httpx.Client
    monkeypatch.setattr(
        bookstore_api, "_create_http_client",
        lambda timeout, trust_env: real_client(
            timeout=timeout, trust_env=trust_env, transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(OrderAPIError, match="仅待支付订单"):
        propose_cancel_order(order_id=5)

# ---- LLM 可见性：真实执行器不出现在工具列表 ----

def test_llm_cannot_see_real_write_executors():
    names = {tool["function"]["name"] for tool in TOOLS}
    assert "propose_order" in names
    assert "propose_cancel_order" in names
    assert "create_order" not in names   # 机制级隔离：LLM 无法直接下单
    assert "cancel_order" not in names


# ---- harness：confirm_request 事件与 run_confirmed ----

def test_confirm_event_extracted_from_tool_result():
    from app.agent.harness import _confirm_event

    card = json.dumps({"ok": True, "action": "confirm_required",
                       "operation_id": "abc", "summary": {"type": "create_order"}})
    event = _confirm_event(card)
    assert event == {"type": "confirm_request", "operation_id": "abc",
                     "summary": {"type": "create_order"}}

    assert _confirm_event(json.dumps({"ok": True})) is None
    assert _confirm_event("not-json") is None


def test_run_confirmed_executes_stored_operation(monkeypatch):
    from app.agent import harness

    captured = {}

    def fake_execute(name, arguments):
        captured["name"] = name
        captured["arguments"] = arguments
        return json.dumps({"ok": True, "order_id": 8})

    def fake_loop(messages):
        captured["injected_user_message"] = messages[-1]["content"]
        yield {"type": "final", "content": "已下单"}

    monkeypatch.setattr(harness, "execute_tool", fake_execute)
    monkeypatch.setattr(harness, "_reply_loop", fake_loop)

    operation = {"op_id": "x" * 16, "tool": "create_order",
                 "arguments": {"items": [{"book_id": 9, "quantity": 2}]}}
    events = list(harness.run_confirmed([], operation))

    # 用存储的原始参数执行，不经 LLM
    assert captured["name"] == "create_order"
    assert json.loads(captured["arguments"]) == operation["arguments"]
    # 执行结果以系统注入的用户消息交给 LLM（不回放合成 tool_calls，兼容思考模式后端）
    assert "order_id" in captured["injected_user_message"]
    assert events[-1]["content"] == "已下单"


def test_run_confirmed_error_after_execution_hints_no_retry(monkeypatch):
    """执行成功但回复生成失败时，错误信息必须提示不要重复操作（防重复下单）。"""
    from app.agent import harness

    monkeypatch.setattr(harness, "execute_tool",
                        lambda name, args: json.dumps({"ok": True, "order_id": 9}))
    monkeypatch.setattr(
        harness, "_reply_loop",
        lambda messages: iter([{"type": "error", "message": "模型返回空响应"}]),
    )

    operation = {"op_id": "y" * 16, "tool": "create_order", "arguments": {"items": []}}
    events = list(harness.run_confirmed([], operation))

    assert "暂勿重复操作" in events[-1]["message"]


# ---- SSE：confirm_request 事件透传 + 确认分支 ----

def test_sse_emits_confirm_request_event(monkeypatch):
    def fake_run(history, message, profile_segment=None, summary=None):
        yield {"type": "delta", "content": "已生成确认卡片"}
        yield {"type": "confirm_request", "operation_id": "abc",
               "summary": {"type": "create_order", "title": "确认创建订单（待支付）"}}
        yield {"type": "final", "content": "请点击卡片确认"}

    monkeypatch.setattr(main_module.harness, "run_agent", fake_run)
    client = TestClient(app)

    response = client.post("/api/v1/agent/chat/stream", json={
        "message": "帮我下单三体", "conversation_id": "sse-confirm-1",
    })

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)
    confirm = [data for name, data in events if name == "confirm_request"][0]
    assert confirm["operation_id"] == "abc"
    assert confirm["summary"]["type"] == "create_order"


def test_sse_confirm_marker_triggers_confirmed_flow(monkeypatch):
    monkeypatch.setattr(write_gate, "get_redis", lambda: None)
    write_gate._memory_store.clear()
    write_gate.set_conversation_id("sse-confirm-2")
    write_gate.create_pending("create_order", {"items": [{"book_id": 9, "quantity": 1}]},
                              {"type": "create_order"})

    def fake_confirmed(history, operation, profile_segment=None, summary=None):
        yield {"type": "final", "content": f"订单已创建（工具={operation['tool']}）"}

    monkeypatch.setattr(main_module.harness, "run_confirmed", fake_confirmed)
    client = TestClient(app)

    response = client.post("/api/v1/agent/chat/stream", json={
        "message": "确认", "conversation_id": "sse-confirm-2",
    })

    events = _parse_sse(response.text)
    assert [name for name, _ in events] == ["done"]
    assert "工具=create_order" in events[0][1]["message"]


def test_sync_chat_confirm_marker(monkeypatch):
    monkeypatch.setattr(write_gate, "get_redis", lambda: None)
    write_gate._memory_store.clear()
    write_gate.set_conversation_id("sync-confirm-1")
    write_gate.create_pending("cancel_order", {"order_id": 5}, {"type": "cancel_order"})

    def fake_model_call_confirmed(history, operation, profile_segment=None, summary=None):
        return "订单已取消"

    monkeypatch.setattr(main_module.harness, "model_call_confirmed", fake_model_call_confirmed)
    client = TestClient(app)

    response = client.post("/api/v1/agent/chat", json={
        "message": "确认", "conversation_id": "sync-confirm-1",
    })

    assert response.json()["message"] == "订单已取消"


def test_stale_confirmation_falls_back_to_normal(monkeypatch):
    # 无待确认操作时，"确认"作为普通消息走正常对话
    called = {}

    def fake_run(history, message, profile_segment=None, summary=None):
        called["message"] = message
        yield {"type": "final", "content": "请问要确认什么？"}

    monkeypatch.setattr(main_module.harness, "run_agent", fake_run)
    monkeypatch.setattr(write_gate, "get_redis", lambda: None)
    write_gate._memory_store.clear()
    client = TestClient(app)

    response = client.post("/api/v1/agent/chat/stream", json={
        "message": "确认", "conversation_id": "stale-1",
    })

    events = _parse_sse(response.text)
    assert events[-1][1]["message"] == "请问要确认什么？"
    assert called["message"] == "确认"


def test_stream_context_survives_across_sse_iterations(monkeypatch):
    """回归：SSE 生成器经线程池逐段迭代，每段从 ASGI 任务重新拷贝上下文。

    会话 ID 必须在 async 端点（任务上下文）内设置；若在生成器内部设置，
    第二次迭代（模拟 LLM 多轮流式输出后的工具执行）读到的会是 None，
    propose 工具将因"缺少 conversation_id"失败。
    """
    seen = {}

    def fake_run(history, message, profile_segment=None, summary=None):
        yield {"type": "delta", "content": "第一段"}   # 迭代边界 1
        seen["cid"] = write_gate.get_conversation_id()  # 迭代边界 2 的线程上下文
        yield {"type": "final", "content": "done"}

    monkeypatch.setattr(main_module.harness, "run_agent", fake_run)
    monkeypatch.setattr(write_gate, "get_redis", lambda: None)
    client = TestClient(app)

    response = client.post("/api/v1/agent/chat/stream", json={
        "message": "hi", "conversation_id": "ctx-regression-1",
    })

    assert response.status_code == 200
    assert seen["cid"] == "ctx-regression-1"
