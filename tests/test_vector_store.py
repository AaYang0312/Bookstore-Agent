# vector_store 单测：过滤表达式组装 / collection 命名随维度切换
from app.config import settings
from app.rag.vector_store import build_filter


def test_build_filter_empty():
    assert build_filter() == ""


def test_build_filter_category():
    assert build_filter(category="科幻") == 'category == "科幻"'


def test_build_filter_price_range():
    assert build_filter(min_price=100, max_price=5000) == "price >= 100 and price <= 5000"


def test_build_filter_combined():
    expr = build_filter(category="科幻", max_price=100)
    assert expr == 'category == "科幻" and price <= 100'


def test_build_filter_escapes_quotes():
    assert build_filter(category='教"育') == 'category == "教\\"育"'


def test_collection_name_embeds_dimension(monkeypatch):
    monkeypatch.setattr(settings, "embedding_dim", 1536)
    assert settings.collection_name == "books_1536"
    monkeypatch.setattr(settings, "embedding_dim", 1024)
    assert settings.collection_name == "books_1024"


def test_rag_ready_requires_milvus_and_model(monkeypatch):
    monkeypatch.setattr(settings, "milvus_uri", None)
    monkeypatch.setattr(settings, "embedding_model_name", "text-embedding-3-small")
    assert settings.rag_ready is False
    monkeypatch.setattr(settings, "milvus_uri", "http://localhost:19530")
    assert settings.rag_ready is True
