# 用户数据工具：画像 / 订单 / 收藏 / 浏览记录（均需 JWT，缺 token 时结构化降级）
import httpx

from app.auth_context import get_token
from app.config import settings

NOT_LOGGED_IN_HINT = "用户未登录（缺少有效的 Authorization token），无法获取个人数据，请基于通用推荐回答并说明原因"


class UserAPIError(Exception):
    """用户数据接口调用异常"""
    pass


def _create_http_client(timeout: float, trust_env: bool) -> httpx.Client:
    return httpx.Client(timeout=timeout, trust_env=trust_env)


def _request_bookstore(path: str, params: dict | None = None) -> dict:
    """携带 JWT 调用 Go 用户接口，复用 book.py 的异常转换约定。"""
    token = get_token()
    if not token:
        raise UserAPIError(NOT_LOGGED_IN_HINT)

    base_url = settings.bookstore_api_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with _create_http_client(timeout=5.0, trust_env=False) as client:
            response = client.get(base_url + path, params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise UserAPIError("payload格式非法")
        if payload.get("code") != 0:
            raise UserAPIError(payload.get("message", "用户接口调用失败"))
    except httpx.TimeoutException:
        raise UserAPIError("用户接口请求超时")
    except httpx.RequestError:
        raise UserAPIError("无法连接 Go 书城服务")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            # token 过期/撤销：与未登录同等的降级语义，避免 500
            raise UserAPIError("用户登录状态已失效，请基于通用推荐回答")
        raise UserAPIError(f"Go 接口返回错误状态: {e.response.status_code}")
    except ValueError:
        raise UserAPIError("Go 接口返回的数据不是合法 JSON")

    return payload


def _validate_paging(page: int, page_size: int) -> None:
    if not type(page) is int or page < 1:
        raise UserAPIError("page 必须为大于等于 1 的整数")
    if not type(page_size) is int or not 1 <= page_size <= 10:
        raise UserAPIError("page_size 必须为 1-10 的整数")


def get_user_profile() -> dict:
    """
    获取当前登录用户的个人资料
    """
    payload = _request_bookstore("/user/profile")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise UserAPIError("接口返回数据格式异常")
    return {
        "ok": True,
        "profile": {
            "user_id": data.get("id"),
            "username": data.get("username", ""),
            "email": data.get("email", ""),
            "phone": data.get("phone", ""),
        },
    }


ORDER_STATUS_TEXT = {0: "待支付", 1: "已支付", 2: "已取消"}


def get_user_orders(page: int = 1, page_size: int = 5) -> dict:
    """
    获取当前登录用户最近的订单列表

    Args:
        page: 页码，默认 1
        page_size: 每页数量，默认 5
    """
    _validate_paging(page, page_size)
    payload = _request_bookstore("/order/list", params={"page": page, "page_size": page_size})
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("orders"), list):
        raise UserAPIError("接口返回数据格式异常")

    orders = []
    for order in data["orders"]:
        if not isinstance(order, dict):
            continue
        items = []
        for item in order.get("order_items") or []:
            if not isinstance(item, dict):
                continue
            book = item.get("book") or {}
            items.append({
                "book_id": item.get("book_id"),
                "title": book.get("title", ""),
                "author": book.get("author", ""),
                "quantity": item.get("quantity", 0),
                "price": item.get("price", 0),
                "subtotal": item.get("subtotal", 0),
            })
        status = order.get("status")
        orders.append({
            "order_id": order.get("id"),
            "order_no": order.get("order_no", ""),
            "total_amount": order.get("total_amount", 0),
            "status": status,
            "status_text": ORDER_STATUS_TEXT.get(status, "未知"),
            "created_at": order.get("created_at", ""),
            "items": items,
        })

    return {
        "ok": True,
        "orders": orders,
        "total": data.get("total", len(orders)),
        "page": data.get("page", page),
        "page_size": data.get("page_size", page_size),
    }


def get_user_favorites(page: int = 1, page_size: int = 5) -> dict:
    """
    获取当前登录用户收藏的图书列表

    Args:
        page: 页码，默认 1
        page_size: 每页数量，默认 5
    """
    _validate_paging(page, page_size)
    payload = _request_bookstore("/favorite/list", params={"page": page, "page_size": page_size})
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("favorites"), list):
        raise UserAPIError("接口返回数据格式异常")

    favorites = []
    for favorite in data["favorites"]:
        if not isinstance(favorite, dict):
            continue
        book = favorite.get("book") or {}
        favorites.append({
            "book_id": book.get("id", favorite.get("book_id")),
            "title": book.get("title", ""),
            "author": book.get("author", ""),
            "type": book.get("type", ""),
            "price": book.get("price", 0),
            "favorited_at": favorite.get("created_at", ""),
        })

    return {
        "ok": True,
        "favorites": favorites,
        "total": data.get("total", len(favorites)),
        "page": page,
        "page_size": page_size,
    }


def get_browse_history(page: int = 1, page_size: int = 5) -> dict:
    """
    获取当前登录用户最近的图书浏览记录

    Args:
        page: 页码，默认 1
        page_size: 每页数量，默认 5
    """
    _validate_paging(page, page_size)
    payload = _request_bookstore("/user/browse-history", params={"page": page, "page_size": page_size})
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise UserAPIError("接口返回数据格式异常")

    records = []
    for record in data["records"]:
        if not isinstance(record, dict):
            continue
        book = record.get("book") or {}
        records.append({
            "book_id": book.get("id", record.get("book_id")),
            "title": book.get("title", ""),
            "author": book.get("author", ""),
            "type": book.get("type", ""),
            "price": book.get("price", 0),
            "viewed_at": record.get("viewed_at", ""),
        })

    return {
        "ok": True,
        "records": records,
        "total": data.get("total", len(records)),
        "page": data.get("page", page),
        "page_size": data.get("page_size", page_size),
    }
