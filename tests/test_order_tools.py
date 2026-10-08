# 订单工具单测：无 JWT 降级 / 异步下单轮询三态 / 幂等键提交 / 业务 message 透传 / 参数校验
import json

import httpx
import pytest

from app.tools import bookstore_api, order as order_module
from app.tools.order import OrderAPIError, cancel_order, create_order, get_order_detail
from app.tools.registry import execute_tool


def _mock_backend(monkeypatch, handler, token="jwt-token-1"):
    monkeypatch.setattr(bookstore_api, "get_token", lambda: token)
    real_client_class = httpx.Client
    monkeypatch.setattr(
        bookstore_api, "_create_http_client",
        lambda timeout, trust_env: real_client_class(
            timeout=timeout, trust_env=trust_env,
            transport=httpx.MockTransport(handler),
        ),
    )


def _fast_poll(monkeypatch, attempts=3):
    monkeypatch.setattr(order_module, "POLL_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(order_module, "POLL_MAX_ATTEMPTS", attempts)


ORDER_PAYLOAD = {
    "code": 0,
    "data": {
        "id": 5, "order_no": "ORD123", "total_amount": 8800,
        "status": 0, "is_paid": False, "created_at": "2026-10-08T10:00:00Z",
        "order_items": [{
            "book_id": 9, "quantity": 2, "price": 4400, "subtotal": 8800,
            "book": {"title": "三体", "author": "刘慈欣"},
        }],
    },
}


def test_order_tools_without_jwt_return_structured_hint(monkeypatch):
    monkeypatch.setattr(bookstore_api, "get_token", lambda: None)

    for name, args in [
        ("create_order", '{"items": [{"book_id": 1, "quantity": 1}]}'),
        ("cancel_order", '{"order_id": 1}'),
        ("get_order_detail", '{"order_id": 1}'),
    ]:
        result = json.loads(execute_tool(name, args))
        assert result["ok"] is False
        assert "用户未登录" in result["error"]


def test_get_order_detail_normalization(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(200, json=ORDER_PAYLOAD)

    _mock_backend(monkeypatch, handler)
    result = get_order_detail(order_id=5)

    assert result["ok"] is True
    order = result["order"]
    assert order["order_id"] == 5
    assert order["status_text"] == "待支付"
    assert order["is_paid"] is False
    assert order["items"][0]["title"] == "三体"
    assert order["items"][0]["subtotal"] == 8800
    assert captured["path"] == "/api/v1/order/5"


def test_get_order_detail_not_found(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"code": -1, "message": "订单不存在"})

    _mock_backend(monkeypatch, handler)
    with pytest.raises(OrderAPIError, match="订单不存在"):
        get_order_detail(order_id=99)


def test_create_order_sync_idempotent_hit(monkeypatch):
    # 受理快查命中幂等键：POST 直接返回 created，无需轮询
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "code": 0, "data": {"status": "created", "order_id": 7, "idempotency_key": "k"},
        })

    _mock_backend(monkeypatch, handler)
    _fast_poll(monkeypatch, attempts=0)
    result = create_order(items=[{"book_id": 9, "quantity": 2}])

    assert result["ok"] is True and result["order_id"] == 7
    # 提交体携带工具生成的 uuid 幂等键与订单项
    assert len(captured["body"]["idempotency_key"]) == 32
    assert captured["body"]["items"] == [{"book_id": 9, "quantity": 2}]


def test_create_order_polls_until_created(monkeypatch):
    _fast_poll(monkeypatch, attempts=5)
    calls = {"submit": 0, "poll": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/order/create"):
            calls["submit"] += 1
            return httpx.Response(202, json={
                "code": 0, "data": {"status": "pending", "idempotency_key": "abc"},
            })
        calls["poll"] += 1
        if calls["poll"] == 1:  # 第一次仍在处理
            return httpx.Response(200, json={"code": 0, "data": {"status": "pending"}})
        return httpx.Response(200, json={
            "code": 0, "data": {"status": "created", "order_id": 8, "order_no": "ORD8"},
        })

    _mock_backend(monkeypatch, handler)
    result = create_order(items=[{"book_id": 9, "quantity": 1}])

    assert result["ok"] is True
    assert result["order_id"] == 8
    assert result["order_no"] == "ORD8"
    assert "待支付" in result["message"]
    assert calls == {"submit": 1, "poll": 2}  # 未重复提交


def test_create_order_failed_reason_from_backend(monkeypatch):
    _fast_poll(monkeypatch, attempts=3)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/order/create"):
            return httpx.Response(202, json={
                "code": 0, "data": {"status": "pending", "idempotency_key": "abc"},
            })
        return httpx.Response(200, json={
            "code": 0, "data": {"status": "failed", "message": "《三体》库存不足"},
        })

    _mock_backend(monkeypatch, handler)
    result = create_order(items=[{"book_id": 9, "quantity": 99}])

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert "库存不足" in result["error"]


def test_create_order_poll_timeout_no_duplicate_hint(monkeypatch):
    _fast_poll(monkeypatch, attempts=2)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/order/create"):
            return httpx.Response(202, json={
                "code": 0, "data": {"status": "pending", "idempotency_key": "abc"},
            })
        return httpx.Response(200, json={"code": 0, "data": {"status": "pending"}})

    _mock_backend(monkeypatch, handler)
    result = create_order(items=[{"book_id": 9, "quantity": 1}])

    assert result["ok"] is False
    assert result["status"] == "pending"
    assert "不要重复下单" in result["error"]
    assert result["idempotency_key"]  # 便于追溯


def test_create_order_mq_unavailable(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"code": -1, "message": "消息队列暂不可用，请稍后重试"})

    _mock_backend(monkeypatch, handler)
    with pytest.raises(OrderAPIError, match="消息队列暂不可用"):
        create_order(items=[{"book_id": 9, "quantity": 1}])


def test_cancel_order_success(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        return httpx.Response(200, json={"code": 0, "message": "订单已取消"})

    _mock_backend(monkeypatch, handler)
    result = cancel_order(order_id=5)

    assert result["ok"] is True
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/v1/order/5/cancel"


def test_cancel_order_conflict_message_preserved(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        # 已支付订单取消：409 + 业务 message
        return httpx.Response(409, json={"code": -1, "message": "仅待支付订单可以取消"})

    _mock_backend(monkeypatch, handler)
    with pytest.raises(OrderAPIError, match="仅待支付订单可以取消"):
        cancel_order(order_id=5)


def test_items_validation(monkeypatch):
    monkeypatch.setattr(bookstore_api, "get_token", lambda: "jwt-token-1")
    bad_inputs = [
        [],
        "not-a-list",
        [{"book_id": 9}],  # 缺 quantity
        [{"book_id": 0, "quantity": 1}],
        [{"book_id": 9, "quantity": 0}],
        [{"book_id": 9, "quantity": 100}],
        [{"book_id": True, "quantity": 1}],
        [{"book_id": 1, "quantity": 1}] * 11,
    ]
    for items in bad_inputs:
        with pytest.raises(OrderAPIError):
            create_order(items=items)

    with pytest.raises(OrderAPIError):
        cancel_order(order_id=0)
    with pytest.raises(OrderAPIError):
        get_order_detail(order_id="5")
