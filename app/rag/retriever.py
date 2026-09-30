# 检索编排：向量召回 →（可选）Rerank 重排 → Top-K 结果组装
import logging

import httpx

from app.config import settings
from app.rag import vector_store
from app.rag.embedder import EmbeddingError, embedder

logger = logging.getLogger(__name__)

SNIPPET_MAX_CHARS = 120


class RetrieverError(Exception):
    """检索编排异常"""
    pass


def _rerank(query: str, documents: list[str], top_n: int) -> list[dict] | None:
    """调用 Rerank API（Cohere/Jina 兼容格式）。失败时返回 None，由上层直通截断。"""
    if not settings.rerank_base_url:
        return None
    headers = {"Content-Type": "application/json"}
    if settings.rerank_api_key is not None:
        headers["Authorization"] = f"Bearer {settings.rerank_api_key.get_secret_value()}"
    payload = {
        "model": settings.rerank_model,
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }
    try:
        with httpx.Client(timeout=settings.rerank_timeout, trust_env=False) as client:
            response = client.post(
                settings.rerank_base_url.rstrip("/") + "/rerank",
                json=payload,
                headers=headers,
            )
        response.raise_for_status()
        body = response.json()
    except Exception as e:
        logger.warning("Rerank 请求失败，回退为向量召回排序：%s", e)
        return None

    results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(results, list):
        logger.warning("Rerank 返回格式异常，回退为向量召回排序")
        return None
    return results


def semantic_search(query: str, category: str | None = None,
                    max_price: int | None = None,
                    min_price: int | None = None,
                    top_k: int | None = None) -> dict:
    """语义检索图书：向量召回 recall_k → 可选重排 → 截断 top_k。

    Returns:
        {"ok": True, "query":..., "results": [...]} 的字典
    """
    top_k = top_k or settings.rag_top_k
    recall_k = max(settings.rag_recall_k, top_k)

    try:
        vector = embedder.embed_query(query)
    except EmbeddingError as e:
        raise RetrieverError(str(e))

    hits = vector_store.search(
        vector,
        top_k=recall_k,
        category=category,
        max_price=max_price,
        min_price=min_price,
    )
    if not hits:
        return {"ok": True, "query": query, "results": [], "message": "向量库中未找到相关图书"}

    # 可选 Rerank：关闭或失败时直接按向量相似度截断
    rerank_scores: dict[int, float] = {}
    if settings.rerank_enabled:
        documents = [f'{h["title"]} {h["author"]} {h["type"]} {h["category"]}' for h in hits]
        reranked = _rerank(query, documents, top_k)
        if reranked:
            order: list[int] = []
            for item in reranked:
                index = item.get("index")
                if isinstance(index, int) and 0 <= index < len(hits):
                    rerank_scores[hits[index]["book_id"]] = item.get("relevance_score", 0.0)
                    order.append(index)
            hits = [hits[i] for i in order] + [h for i, h in enumerate(hits) if i not in set(order)]

    results = []
    for hit in hits[:top_k]:
        results.append({
            "book_id": hit["book_id"],
            "title": hit["title"],
            "author": hit["author"],
            "type": hit["type"],
            "category": hit["category"],
            "price": hit["price"],
            "stock": hit["stock"],
            "snippet": (hit.get("description") or "")[:SNIPPET_MAX_CHARS],
            "score": round(float(rerank_scores.get(hit["book_id"], hit["score"])), 4),
            "reranked": hit["book_id"] in rerank_scores,
        })
    return {"ok": True, "query": query, "results": results}
