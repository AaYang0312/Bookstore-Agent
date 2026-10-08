# 收藏写工具：新增 / 移除 / 查询收藏状态（均需 JWT，缺 token 时结构化降级）
# 注意：书城的新增/移除收藏走 Kafka 异步受理，接口返回“已受理”而非最终结果；
# 需要确认最终状态时用 check_favorite 复核
from app.tools.bookstore_api import BookstoreAPIError, request_bookstore


class FavoriteAPIError(BookstoreAPIError):
    """收藏接口调用异常"""
    pass


def _call(method: str, path: str) -> dict:
    return request_bookstore(method, path, error_class=FavoriteAPIError)


def _validate_book_id(book_id) -> int:
    # type() is int 同时排除 bool（LLM 可能传 true/false）
    if type(book_id) is not int or book_id < 1:
        raise FavoriteAPIError("book_id 必须为大于等于 1 的整数")
    return book_id


def add_favorite(book_id: int) -> dict:
    """收藏一本图书（异步受理，稍后生效）

    Args:
        book_id: 图书 ID，必须来自检索工具结果
    """
    book_id = _validate_book_id(book_id)
    _call("POST", f"/favorite/{book_id}")
    return {
        "ok": True,
        "book_id": book_id,
        "message": "收藏请求已受理（异步生效，稍后可在收藏列表查看）",
    }


def remove_favorite(book_id: int) -> dict:
    """移除一本图书的收藏（异步受理，稍后生效）

    Args:
        book_id: 图书 ID
    """
    book_id = _validate_book_id(book_id)
    _call("DELETE", f"/favorite/{book_id}")
    return {
        "ok": True,
        "book_id": book_id,
        "message": "移除收藏请求已受理（异步生效，稍后可在收藏列表查看）",
    }


def check_favorite(book_id: int) -> dict:
    """查询当前用户是否已收藏某本图书

    Args:
        book_id: 图书 ID
    """
    book_id = _validate_book_id(book_id)
    payload = _call("GET", f"/favorite/{book_id}/check")
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("is_favorited"), bool):
        raise FavoriteAPIError("接口返回数据格式异常")
    return {
        "ok": True,
        "book_id": book_id,
        "is_favorited": data["is_favorited"],
    }
