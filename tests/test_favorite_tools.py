# 收藏写工具单测：无 JWT 降级 / 方法与路径 / 异步受理语义 / 业务 message 透传 / 参数校验
import json

import httpx
import pytest

from app.tools import bookstore_api
from app.tools.favorite import FavoriteAPIError, add_favorite, check_favorite, remove_favorite
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


def test_favorite_tools_without_jwt_return_structured_hint(monkeypatch):
    monkeypatch.setattr(bookstore_api, "get_token", lambda: None)

    for name in ["add_favorite", "remove_favorite", "check_favorite"]:
        result = json.loads(execute_tool(name, '{"book_id": 1}'))
        assert result["ok"] is False
        assert "用户未登录" in result["error"]


def test_add_favorite_sends_post_and_reports_async(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"code": 0, "message": "收藏请求已受理"})

    _mock_backend(monkeypatch, handler)
    result = add_favorite(book_id=3)

    assert result["ok"] is True
    assert "已受理" in result["message"]  # 异步语义：受理而非完成
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/v1/favorite/3"
    assert captured["authorization"] == "Bearer jwt-token-1"


def test_remove_favorite_sends_delete(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        return httpx.Response(200, json={"code": 0, "message": "移除收藏请求已受理"})

    _mock_backend(monkeypatch, handler)
    result = remove_favorite(book_id=3)

    assert result["ok"] is True
    assert captured["method"] == "DELETE"
    assert captured["path"] == "/api/v1/favorite/3"


def test_check_favorite_parses_status(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(200, json={"code": 0, "data": {"is_favorited": True}})

    _mock_backend(monkeypatch, handler)
    result = check_favorite(book_id=9)

    assert result == {"ok": True, "book_id": 9, "is_favorited": True}
    assert captured["path"] == "/api/v1/favorite/9/check"


def test_backend_business_message_preserved_on_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        # Go 侧 MQ 不可用时返回 500 + 业务 message
        return httpx.Response(500, json={"code": -1, "message": "收藏服务繁忙，请稍后重试"})

    _mock_backend(monkeypatch, handler)
    with pytest.raises(FavoriteAPIError, match="收藏服务繁忙"):
        add_favorite(book_id=1)


def test_expired_token_degrades(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"code": -1, "message": "无效的token"})

    _mock_backend(monkeypatch, handler)
    with pytest.raises(FavoriteAPIError, match="登录状态已失效"):
        add_favorite(book_id=1)


def test_book_id_validation(monkeypatch):
    monkeypatch.setattr(bookstore_api, "get_token", lambda: "jwt-token-1")
    for bad in [0, -1, True, "3", None, 1.5]:
        with pytest.raises(FavoriteAPIError):
            add_favorite(book_id=bad)
