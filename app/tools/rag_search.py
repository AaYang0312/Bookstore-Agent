# 语义检索图书工具（RAG 入口）
from app.config import settings
from app.rag.retriever import RetrieverError, semantic_search


class RAGSearchError(Exception):
    """语义检索工具异常"""
    pass


def rag_available() -> bool:
    """RAG 是否已启用（Milvus + Embedding 均已配置）。"""
    return settings.rag_ready


def semantic_search_books(query: str, category: str | None = None,
                          max_price: int | None = None) -> dict:
    """
    语义搜索图书：按自然语言描述在书城向量知识库中检索相关图书。

    Args:
        query: 自然语言描述或模糊需求（必填）
        category: 可选，限定分类名称
        max_price: 可选，价格上限（元）
    Returns:
        包含语义匹配图书列表的字典
    """
    if not isinstance(query, str):
        raise RAGSearchError("查询描述格式不正确")
    query = query.strip()
    if not query:
        raise RAGSearchError("查询描述不能为空")
    if len(query) > 200:
        raise RAGSearchError("查询描述长度不能超过 200 个字符")

    if category is not None:
        if not isinstance(category, str):
            raise RAGSearchError("分类格式不正确")
        category = category.strip()
        if not category:
            category = None
        elif len(category) > 50:
            raise RAGSearchError("分类名称不能超过 50 个字符")

    if max_price is not None:
        if not type(max_price) is int:
            raise RAGSearchError("max_price 必须为整数")
        if max_price < 0:
            raise RAGSearchError("max_price 不能为负数")

    if not rag_available():
        raise RAGSearchError("语义检索未启用，请改用 search_books 按关键词搜索")

    try:
        return semantic_search(query=query, category=category, max_price=max_price)
    except RetrieverError as e:
        raise RAGSearchError(f"语义检索失败：{e}")
