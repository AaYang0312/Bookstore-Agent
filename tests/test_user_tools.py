# 用户数据工具单测：无 JWT 结构化降级 / 带 JWT 透传 / 401 降级
import json

import httpx
import pytest

from app.tools import user as user_module
from app.tools.registry import execute_tool
from app.tools.user import NOT_LOGGED_IN_HINT, UserAPIError


def test_tools_without_jwt_return_structured_hint(monkeypatch):
    monkeypatch.setattr(user_module, "get_token", lambda: None)

    for name in ["get_user_profile", "get_user_orders", "get_user_favorites", "get_browse_history"]:
        result = json.loads(execute_tool(name, "{}"))
        assert result["ok"] is False
        assert "用户未登录" in result["error"]


def test_get_user_profile_with_jwt(monkeypatch):
    monkeypatch.setattr(user_module, "get_token", lambda: "jwt-token-1")

    payload = {
        "code": 0,
        "message": "ok",
        "data": {"id": 7, "username": "小明", "email": "x@test.com", "phone": "123"},
    }
    captured = {}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("Authorization")
        return httpx.Response(200, json=payload)

    real_client_class = httpx.Client
    monkeypatch.setattr(
        user_module, "_create_http_client",
        lambda timeout, trust_env: real_client_class(
            timeout=timeout, trust_env=trust_env,
            transport=httpx.MockTransport(mock_handler),
        ),
    )

    result = user_module.get_user_profile()

    assert result["ok"] is True
    assert result["profile"]["username"] == "小明"
    assert captured["path"] == "/api/v1/user/profile"
    assert captured["authorization"] == "Bearer jwt-token-1"  # JWT 原样透传


def test_get_user_orders_normalization(monkeypatch):
    monkeypatch.setattr(user_module, "get_token", lambda: "jwt-token-1")
    payload = {
        "code": 0,
        "data": {
            "orders": [{
                "id": 3, "order_no": "ORD1", "total_amount": 100,
                "status": 1, "created_at": "2026-09-30",
                "order_items": [{
                    "book_id": 9, "quantity": 2, "price": 40, "subtotal": 80,
                    "book": {"title": "三体", "author": "刘慈欣"},
                }],
            }],
            "total": 1, "page": 1, "page_size": 5,
        },
    }

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/order/list"
        return httpx.Response(200, json=payload)

    real_client_class = httpx.Client
    monkeypatch.setattr(
        user_module, "_create_http_client",
        lambda timeout, trust_env: real_client_class(
            timeout=timeout, trust_env=trust_env,
            transport=httpx.MockTransport(mock_handler),
        ),
    )

    result = user_module.get_user_orders()
    assert result["ok"] is True
    assert result["orders"][0]["status_text"] == "已支付"
    assert result["orders"][0]["items"][0]["title"] == "三体"


def test_expired_token_degrades_not_500(monkeypatch):
    monkeypatch.setattr(user_module, "get_token", lambda: "expired-token")

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"code": -1, "message": "无效的token"})

    real_client_class = httpx.Client
    monkeypatch.setattr(
        user_module, "_create_http_client",
        lambda timeout, trust_env: real_client_class(
            timeout=timeout, trust_env=trust_env,
            transport=httpx.MockTransport(mock_handler),
        ),
    )

    with pytest.raises(UserAPIError, match="登录状态已失效"):
        user_module.get_user_profile()


def test_paging_validation(monkeypatch):
    monkeypatch.setattr(user_module, "get_token", lambda: "jwt-token-1")
    with pytest.raises(UserAPIError, match="page"):
        user_module.get_user_orders(page=0)
    with pytest.raises(UserAPIError, match="page_size"):
        user_module.get_user_favorites(page_size=99)
