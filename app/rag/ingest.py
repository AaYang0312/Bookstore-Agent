# 入库管道：拉取 Go 图书 → 组装文档 → 向量化 → 写 Milvus（全量/增量）
"""
用法：
    python -m app.rag.ingest --full           # 全量重灌（含陈旧记录清理）
    python -m app.rag.ingest --incremental    # 按 updated_at 水位增量
    python -m app.rag.ingest --incremental --since "2026-01-01T00:00:00+08:00"
"""
import argparse
import logging
import sys
from datetime import datetime, timezone

import httpx

from app.config import settings
from app.rag import vector_store
from app.rag.embedder import EmbeddingError, embedder
from app.redis_client import get_redis

logger = logging.getLogger(__name__)

DESCRIPTION_MAX_CHARS = 1000

WATERMARK_KEY = "rag:ingest:watermark"  # 按维度（collection）区分水位


class IngestError(Exception):
    """入库管道异常"""
    pass


def _internal_sync_url() -> str:
    """内部只读同步接口地址：与 /api/v1 同级的 /internal/books/sync。"""
    base = settings.bookstore_api_base_url.rstrip("/")
    suffix = "/api/v1"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    return base + "/internal/books/sync"


def _fetch_sync_page(since: str | None, page: int, page_size: int) -> dict:
    """调用 Go 内部只读分页接口拉取图书（带共享密钥，不走限流的 /book/search）。"""
    if settings.internal_sync_secret is None:
        raise IngestError("INTERNAL_SYNC_SECRET 未配置，无法从 Go 后端拉取图书数据")
    params: dict = {"page": page, "page_size": page_size}
    if since:
        params["since"] = since
    headers = {"X-Internal-Token": settings.internal_sync_secret.get_secret_value()}
    try:
        with httpx.Client(timeout=15.0, trust_env=False) as client:
            response = client.get(_internal_sync_url(), params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()
    except httpx.TimeoutException:
        raise IngestError("拉取图书数据超时")
    except httpx.RequestError:
        raise IngestError("无法连接 Go 书城服务")
    except ValueError:
        raise IngestError("Go 同步接口返回的数据不是合法 JSON")

    if not isinstance(payload, dict) or payload.get("code") != 0:
        message = payload.get("message", "未知错误") if isinstance(payload, dict) else "响应格式异常"
        raise IngestError(f"Go 同步接口返回错误：{message}")

    data = payload.get("data")
    if not isinstance(data, dict):
        raise IngestError("Go 同步接口返回数据格式异常")
    return data


def build_document(book: dict, category_name: str) -> str:
    """组装用于向量化的图书文档：多字段拼接提高语义区分度。"""
    parts = [
        str(book.get("title") or ""),
        str(book.get("author") or ""),
        str(book.get("publisher") or ""),
        str(book.get("type") or ""),
        category_name,
    ]
    description = str(book.get("description") or "")[:DESCRIPTION_MAX_CHARS]
    parts.append(description)
    return " | ".join(p.strip() for p in parts if p and p.strip())


def build_record(book: dict, vector: list[float], category_name: str) -> dict:
    def _to_int(value, default: int = 0) -> int:
        return value if isinstance(value, int) else default

    updated_at = book.get("updated_at") or ""
    try:
        ts = int(
            datetime.fromisoformat(updated_at).astimezone(timezone.utc).timestamp()
        )
    except (ValueError, TypeError):
        ts = 0
    return {
        "book_id": _to_int(book.get("id")),
        "vector": vector,
        "title": str(book.get("title") or "")[:500],
        "author": str(book.get("author") or "")[:200],
        "type": str(book.get("type") or "")[:100],
        "category": category_name[:100],
        "category_id": _to_int(book.get("category_id")),
        "price": _to_int(book.get("price")),
        "stock": _to_int(book.get("stock")),
        "status": _to_int(book.get("status"), 1),
        "updated_at": ts,
        "description": str(book.get("description") or "")[:1000],
    }


def _embed_and_write(books: list[dict]) -> int:
    """批量组装文档 → 向量化 → upsert，返回成功写入数量。"""
    if not books:
        return 0
    docs = [
        build_document(book, str(book.get("category_name") or ""))
        for book in books
    ]
    vectors = embedder.embed_texts(docs)
    if len(vectors) != len(books):
        raise IngestError("向量化返回数量与输入不一致")
    records = [
        build_record(book, vec, str(book.get("category_name") or ""))
        for book, vec in zip(books, vectors)
    ]
    return vector_store.upsert_books(records)


def _valid_next_since(value) -> str | None:
    """过滤无效水位：空串或零值时间（0001-01-01）不采纳，避免水位回退导致全量重灌。"""
    if not isinstance(value, str) or not value or value.startswith("0001-"):
        return None
    return value


def _iter_pages(since: str | None):
    """按页拉取图书，产出 (books, page_data)。"""
    page = 1
    while True:
        data = _fetch_sync_page(since, page, settings.ingest_page_size)
        books = data.get("books")
        if books is None:
            books = []  # Go 空结果可能返回 null，按空列表处理
        elif not isinstance(books, list):
            raise IngestError("Go 同步接口的 books 不是列表")
        yield books, data
        total_pages = data.get("total_pages")
        if not isinstance(total_pages, int) or page >= total_pages or not books:
            return
        page += 1


def _get_watermark() -> str | None:
    client = get_redis()
    if client is None:
        return None
    try:
        value = client.get(f"{WATERMARK_KEY}:{settings.collection_name}")
        return value if isinstance(value, str) and value else None
    except Exception:
        return None


def _set_watermark(since: str) -> None:
    client = get_redis()
    if client is None:
        return
    try:
        client.set(f"{WATERMARK_KEY}:{settings.collection_name}", since)
    except Exception:
        logger.warning("写入入库水位失败（不影响本次数据）")


def ingest_full() -> dict:
    """全量入库：逐页拉取并 upsert，结束后清理已不在数据库/已下架的记录。"""
    if not settings.rag_ready:
        raise IngestError("RAG 未启用（需配置 MILVUS_URI 与 EMBEDDING_MODEL_NAME）")

    vector_store.ensure_collection()
    written = 0
    active_ids: set[int] = set()
    inactive_ids: set[int] = set()
    for books, _data in _iter_pages(since=None):
        active = [b for b in books if isinstance(b, dict) and b.get("status") == 1]
        inactive = [b for b in books if isinstance(b, dict) and b.get("status") != 1]
        written += _embed_and_write(active)
        for b in active:
            if isinstance(b.get("id"), int):
                active_ids.add(b["id"])
        for b in inactive:
            if isinstance(b.get("id"), int):
                inactive_ids.add(b["id"])

    # 清理 Milvus 中已不在库或已下架的记录
    stale = set(vector_store.list_book_ids()) - active_ids - inactive_ids
    deleted = vector_store.delete_books(sorted(stale | inactive_ids))
    logger.info("全量入库完成：写入 %d 本，清理 %d 条", written, deleted)
    return {"written": written, "deleted": deleted}


def ingest_incremental(since: str | None = None) -> dict:
    """增量入库：按 updated_at 水位拉取变更书籍，仅重灌变更部分。"""
    if not settings.rag_ready:
        raise IngestError("RAG 未启用（需配置 MILVUS_URI 与 EMBEDDING_MODEL_NAME）")

    watermark = since or _get_watermark()
    if not watermark:
        raise IngestError(
            "未找到入库水位（Redis 未配置或首次运行），请先执行 --full 或用 --since 指定起点"
        )

    vector_store.ensure_collection()
    written = 0
    deleted = 0
    next_since: str | None = None
    for books, data in _iter_pages(since=watermark):
        active = [b for b in books if isinstance(b, dict) and b.get("status") == 1]
        inactive = [b for b in books if isinstance(b, dict) and b.get("status") != 1]
        written += _embed_and_write(active)
        if inactive:
            ids = [b["id"] for b in inactive if isinstance(b.get("id"), int)]
            deleted += vector_store.delete_books(ids)
        candidate = _valid_next_since(data.get("next_since"))
        if candidate:
            next_since = candidate

    if next_since:
        _set_watermark(next_since)
    logger.info("增量入库完成（since=%s）：写入 %d 本，删除 %d 条", watermark, written, deleted)
    return {"since": watermark, "written": written, "deleted": deleted}


def scheduled_incremental() -> None:
    """APScheduler 定时任务入口：增量失败只记录日志，不影响服务。"""
    try:
        result = ingest_incremental()
        logger.info("定时增量入库完成：写入 %d 本", result.get("written", 0))
    except Exception as e:
        logger.warning("定时增量入库失败：%s", e)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="图书向量库入库管道")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--full", action="store_true", help="全量重灌")
    group.add_argument("--incremental", action="store_true", help="按水位增量")
    parser.add_argument("--since", type=str, default=None, help="增量起始 updated_at（ISO 格式）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        if args.full:
            result = ingest_full()
        else:
            result = ingest_incremental(since=args.since)
    except (IngestError, EmbeddingError, vector_store.VectorStoreError) as e:
        print(f"入库失败：{e}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
