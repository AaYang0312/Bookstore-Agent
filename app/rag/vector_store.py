# Milvus 向量库封装：collection 建/删、upsert、删除、混合检索（ANN + 标量过滤）
import logging

from app.config import settings

logger = logging.getLogger(__name__)

try:
    from pymilvus import DataType, MilvusClient
    PYMILVUS_AVAILABLE = True
except ImportError:  # pragma: no cover - 依赖缺失时仅影响 RAG
    MilvusClient = None
    DataType = None
    PYMILVUS_AVAILABLE = False


class VectorStoreError(Exception):
    """向量库操作异常"""
    pass


SCALAR_OUTPUT_FIELDS = [
    "book_id", "title", "author", "type", "category",
    "category_id", "price", "stock", "status", "updated_at", "description",
]

_client = None


def get_client():
    """惰性创建全局 MilvusClient 单例。"""
    global _client
    if not PYMILVUS_AVAILABLE:
        raise VectorStoreError("pymilvus 未安装")
    if not settings.milvus_uri:
        raise VectorStoreError("MILVUS_URI 未配置")
    if _client is None:
        try:
            _client = MilvusClient(uri=settings.milvus_uri)
        except Exception as e:
            raise VectorStoreError(f"连接 Milvus 失败：{e}")
    return _client


def reset_client():
    """测试用：重置单例。"""
    global _client
    _client = None


def build_filter(category: str | None = None, max_price: int | None = None,
                 min_price: int | None = None) -> str:
    """组装标量过滤表达式（分类、价格区间）。"""
    conditions = []
    if category:
        escaped = category.replace('"', '\\"')
        conditions.append(f'category == "{escaped}"')
    if min_price is not None:
        conditions.append(f"price >= {int(min_price)}")
    if max_price is not None:
        conditions.append(f"price <= {int(max_price)}")
    return " and ".join(conditions)


def ensure_collection() -> str:
    """确保 collection 存在（不存在则按当前维度建表 + HNSW 索引），返回名称。"""
    client = get_client()
    name = settings.collection_name
    try:
        if client.has_collection(name):
            return name
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("book_id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=settings.embedding_dim)
        schema.add_field("title", DataType.VARCHAR, max_length=500)
        schema.add_field("author", DataType.VARCHAR, max_length=200)
        schema.add_field("type", DataType.VARCHAR, max_length=100)
        schema.add_field("category", DataType.VARCHAR, max_length=100)
        schema.add_field("category_id", DataType.INT64)
        schema.add_field("price", DataType.INT64)
        schema.add_field("stock", DataType.INT64)
        schema.add_field("status", DataType.INT64)
        schema.add_field("updated_at", DataType.INT64)
        schema.add_field("description", DataType.VARCHAR, max_length=1024)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="IP",  # 向量已 L2 归一化，IP 等价余弦相似度
            params={"M": 16, "efConstruction": 200},
        )
        client.create_collection(name, schema=schema, index_params=index_params)
        logger.info("已创建 Milvus collection %s（dim=%d）", name, settings.embedding_dim)
        return name
    except VectorStoreError:
        raise
    except Exception as e:
        raise VectorStoreError(f"创建 Milvus collection 失败：{e}")


def drop_collection() -> None:
    client = get_client()
    name = settings.collection_name
    try:
        if client.has_collection(name):
            client.drop_collection(name)
            logger.info("已删除 Milvus collection %s", name)
    except Exception as e:
        raise VectorStoreError(f"删除 Milvus collection 失败：{e}")


def upsert_books(records: list[dict]) -> int:
    """批量 upsert 图书向量记录，返回写入数量。"""
    if not records:
        return 0
    client = get_client()
    name = ensure_collection()
    try:
        client.upsert(collection_name=name, data=records)
        return len(records)
    except Exception as e:
        raise VectorStoreError(f"写入 Milvus 失败：{e}")


def delete_books(book_ids: list[int]) -> int:
    """按 book_id 删除向量记录（下架书清理）。"""
    if not book_ids:
        return 0
    client = get_client()
    name = ensure_collection()
    try:
        client.delete(collection_name=name, ids=[int(i) for i in book_ids])
        return len(book_ids)
    except Exception as e:
        raise VectorStoreError(f"删除 Milvus 记录失败：{e}")


def list_book_ids(batch_size: int = 4096) -> list[int]:
    """分批遍历 collection 内全部 book_id，供全量入库时比对陈旧记录。"""
    client = get_client()
    name = settings.collection_name
    if not client.has_collection(name):
        return []
    ids: list[int] = []
    cursor = 0
    try:
        while True:
            rows = client.query(
                collection_name=name,
                filter=f"book_id >= {cursor}",
                output_fields=["book_id"],
                limit=batch_size,
            )
            if not rows:
                break
            for row in rows:
                ids.append(int(row["book_id"]))
            if len(rows) < batch_size:
                break
            cursor = max(ids) + 1
        return ids
    except Exception as e:
        raise VectorStoreError(f"查询 Milvus book_id 失败：{e}")


def _hit_field(hit, key, default=None):
    """兼容 pymilvus 的 Hit 对象与普通 dict 两种返回形态取字段。"""
    if isinstance(hit, dict):
        return hit.get(key, default)
    try:
        return hit[key]
    except (KeyError, IndexError, TypeError):
        return default


def search(vector: list[float], top_k: int,
           category: str | None = None, max_price: int | None = None,
           min_price: int | None = None) -> list[dict]:
    """向量检索 + 标量过滤，返回按相似度排序的结果列表。"""
    client = get_client()
    name = settings.collection_name
    if not client.has_collection(name):
        raise VectorStoreError("向量库尚未初始化，请先执行入库（python -m app.rag.ingest --full）")
    expr = build_filter(category, max_price, min_price)
    try:
        response = client.search(
            collection_name=name,
            data=[vector],
            limit=top_k,
            filter=expr if expr else "",
            output_fields=SCALAR_OUTPUT_FIELDS,
            search_params={"metric_type": "IP", "params": {"ef": max(64, top_k * 4)}},
        )
    except Exception as e:
        raise VectorStoreError(f"Milvus 检索失败：{e}")

    hits = response[0] if response else []
    results = []
    for hit in hits:
        entity = _hit_field(hit, "entity", {}) or {}
        results.append({
            "book_id": entity.get("book_id", _hit_field(hit, "book_id", _hit_field(hit, "id"))),
            "title": entity.get("title", ""),
            "author": entity.get("author", ""),
            "type": entity.get("type", ""),
            "category": entity.get("category", ""),
            "category_id": entity.get("category_id", 0),
            "price": entity.get("price", 0),
            "stock": entity.get("stock", 0),
            "status": entity.get("status", 1),
            "updated_at": entity.get("updated_at", 0),
            "description": entity.get("description", ""),
            "score": _hit_field(hit, "distance", 0.0),
        })
    return results


def count() -> int | None:
    """返回 collection 内记录数，失败或不存在时返回 None。"""
    try:
        client = get_client()
        name = settings.collection_name
        if not client.has_collection(name):
            return 0
        stats = client.get_collection_stats(name)
        return int(stats.get("row_count", 0))
    except Exception:
        return None


def ready() -> bool:
    """连通性探测，供 /health 使用。"""
    try:
        get_client().get_server_version()
        return True
    except Exception:
        return False
