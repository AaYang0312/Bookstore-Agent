# retriever 单测：召回 → 截断 / Rerank 重排 / 空结果
from app.rag import retriever


def _hit(book_id, title, score):
    return {
        "book_id": book_id, "title": title, "author": "作者", "type": "科幻",
        "category": "科幻小说", "category_id": 1, "price": 59, "stock": 10,
        "status": 1, "updated_at": 1700000000, "description": "一段关于宇宙文明的描述",
        "score": score,
    }


def test_semantic_search_truncates_to_top_k(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "rag_top_k", 3)
    monkeypatch.setattr(settings, "rag_recall_k", 5)
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(retriever.embedder, "embed_query", lambda q: [0.1, 0.2])
    monkeypatch.setattr(
        retriever.vector_store, "search",
        lambda vector, top_k, **kwargs: [_hit(i, f"书{i}", 0.9 - i * 0.1) for i in range(top_k)],
    )

    result = retriever.semantic_search("想找讲宇宙文明的科幻小说")

    assert result["ok"] is True
    assert len(result["results"]) == 3
    assert result["results"][0]["book_id"] == 0
    assert "snippet" in result["results"][0]


def test_semantic_search_passes_scalar_filters(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(retriever.embedder, "embed_query", lambda q: [0.1, 0.2])
    captured = {}

    def fake_search(vector, top_k, category=None, max_price=None, min_price=None):
        captured.update(category=category, max_price=max_price, top_k=top_k)
        return [_hit(1, "三体", 0.95)]

    monkeypatch.setattr(retriever.vector_store, "search", fake_search)

    retriever.semantic_search("科幻", category="科幻", max_price=100)

    assert captured["category"] == "科幻"
    assert captured["max_price"] == 100


def test_semantic_search_rerank_reorders(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rerank_base_url", "http://rerank.test")
    monkeypatch.setattr(settings, "rag_top_k", 2)
    monkeypatch.setattr(settings, "rag_recall_k", 3)
    monkeypatch.setattr(retriever.embedder, "embed_query", lambda q: [0.1, 0.2])
    monkeypatch.setattr(
        retriever.vector_store, "search",
        lambda vector, top_k, **kwargs: [_hit(i, f"书{i}", 0.9 - i * 0.1) for i in range(top_k)],
    )
    # Rerank 认为 index=1 最相关
    monkeypatch.setattr(
        retriever, "_rerank",
        lambda query, documents, top_n: [
            {"index": 1, "relevance_score": 0.98},
            {"index": 0, "relevance_score": 0.42},
        ],
    )

    result = retriever.semantic_search("模糊找书")

    assert result["results"][0]["book_id"] == 1
    assert result["results"][0]["reranked"] is True
    assert result["results"][0]["score"] == 0.98


def test_semantic_search_rerank_failure_falls_back(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rerank_base_url", "http://rerank.test")
    monkeypatch.setattr(settings, "rag_top_k", 2)
    monkeypatch.setattr(retriever.embedder, "embed_query", lambda q: [0.1, 0.2])
    monkeypatch.setattr(
        retriever.vector_store, "search",
        lambda vector, top_k, **kwargs: [_hit(i, f"书{i}", 0.9 - i * 0.1) for i in range(top_k)],
    )
    # Rerank 服务不可用：返回 None，回退向量排序
    monkeypatch.setattr(retriever, "_rerank", lambda query, documents, top_n: None)

    result = retriever.semantic_search("模糊找书")

    assert result["ok"] is True
    assert result["results"][0]["book_id"] == 0
    assert result["results"][0]["reranked"] is False


def test_semantic_search_empty_results(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(retriever.embedder, "embed_query", lambda q: [0.1, 0.2])
    monkeypatch.setattr(retriever.vector_store, "search", lambda vector, top_k, **kwargs: [])

    result = retriever.semantic_search("冷门需求")

    assert result["ok"] is True
    assert result["results"] == []
