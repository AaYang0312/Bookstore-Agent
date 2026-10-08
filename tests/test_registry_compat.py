# 工具注册回归：未配置 RAG 时现有三工具行为不变；配置后新工具注册
import importlib
import json

from app.config import settings
from app.tools import registry


def test_base_tools_always_registered():
    names = {tool["function"]["name"] for tool in registry.TOOLS}
    # 现有三工具必须始终存在
    assert {"search_books", "get_book_detail", "search_web"} <= names
    # 用户数据四工具随注册表提供（无 JWT 时结构化降级，而非不注册）
    assert {"get_user_profile", "get_user_orders", "get_user_favorites",
            "get_browse_history"} <= names
    assert set(registry.TOOL_MAP) >= names


def test_semantic_tool_not_registered_without_rag(monkeypatch):
    # 本地 .env 未配置 Milvus：semantic_search_books 不应出现在工具列表
    monkeypatch.setattr(settings, "milvus_uri", None)
    monkeypatch.setattr(settings, "embedding_model_name", None)
    assert settings.rag_ready is False

    names = {tool["function"]["name"] for tool in registry.TOOLS}
    assert "semantic_search_books" not in names


def test_semantic_tool_registered_when_rag_configured(monkeypatch):
    monkeypatch.setattr(settings, "milvus_uri", "http://localhost:19530")
    monkeypatch.setattr(settings, "embedding_model_name", "text-embedding-test")
    assert settings.rag_ready is True

    try:
        reloaded = importlib.reload(registry)
        names = {tool["function"]["name"] for tool in reloaded.TOOLS}
        assert "semantic_search_books" in names
        assert "semantic_search_books" in reloaded.TOOL_MAP
        # 3 基础 + 1 语义 + 4 用户数据 + 3 收藏 + 3 订单
        assert len(names) == 14
    finally:
        importlib.reload(registry)  # 恢复未配置状态


def test_semantic_tool_degrades_without_rag(monkeypatch):
    from app.tools.rag_search import RAGSearchError, semantic_search_books

    monkeypatch.setattr(settings, "milvus_uri", None)
    monkeypatch.setattr(settings, "embedding_model_name", None)

    result = json.loads(registry.execute_tool(
        "semantic_search_books", '{"query": "科幻"}'))
    assert result["ok"] is False
    assert "search_books" in result["error"]  # 提示退回关键词搜索


def test_semantic_tool_param_validation():
    from app.tools.rag_search import RAGSearchError, semantic_search_books
    import pytest

    with pytest.raises(RAGSearchError):
        semantic_search_books(query="   ")
    with pytest.raises(RAGSearchError):
        semantic_search_books(query="x" * 201)
    with pytest.raises(RAGSearchError):
        semantic_search_books(query="ok", max_price=-1)


def test_execute_tool_unknown_and_bad_arguments():
    result = json.loads(registry.execute_tool("no_such_tool", "{}"))
    assert result["ok"] is False and "未知工具" in result["error"]

    result = json.loads(registry.execute_tool("search_books", "not-json"))
    assert result["ok"] is False and "JSON" in result["error"]

    result = json.loads(registry.execute_tool("search_books", "[1,2]"))
    assert result["ok"] is False and "对象" in result["error"]
