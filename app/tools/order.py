# 订单工具：代创建订单（待支付，不代支付）/ 取消待支付订单 / 订单详情
# 下单为 Kafka 异步：POST /order/create 受理后返回 pending，本模块内部轮询
# /order/create/result 轮换为最终三态（created/failed/超时），对 LLM 呈现同步语义
import logging
import time
import uuid

from app.tools.bookstore_api import BookstoreAPIError, request_bookstore

logger = logging.getLogger(__name__)

# 轮询参数：0.5s 间隔 × 20 次 = 最长等待约 10s（测试中会调小）
POLL_INTERVAL_SECONDS = 0.5
POLL_MAX_ATTEMPTS = 20

MAX_ITEMS_PER_ORDER = 10
MAX_QUANTITY_PER_ITEM = 99

ORDER_STATUS_TEXT = {0: "待支付", 1: "已支付", 2: "已取消"}


class OrderAPIError(BookstoreAPIError):
    """订单接口调用异常"""
    pass


def _call(method: str, path: str, params: dict | None = None,
          json_body: dict | None = None) -> dict:
    return request_bookstore(method, path, params=params, json_body=json_body,
                             error_class=OrderAPIError)


def _validate_order_id(order_id) -> int:
    if type(order_id) is not int or order_id < 1:
        raise OrderAPIError("order_id 必须为大于等于 1 的整数")
    return order_id


def _validate_items(items) -> list[dict]:
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_ITEMS_PER_ORDER:
        raise OrderAPIError(f"items 必须为 1-{MAX_ITEMS_PER_ORDER} 个订单项的数组")
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            raise OrderAPIError("每个订单项必须是包含 book_id 与 quantity 的对象")
        book_id = item.get("book_id")
        quantity = item.get("quantity")
        if type(book_id) is not int or book_id < 1:
            raise OrderAPIError("每个订单项的 book_id 必须为大于等于 1 的整数")
        if type(quantity) is not int or not 1 <= quantity <= MAX_QUANTITY_PER_ITEM:
            raise OrderAPIError(f"购买数量必须为 1-{MAX_QUANTITY_PER_ITEM} 的整数")
        normalized.append({"book_id": book_id, "quantity": quantity})
    return normalized


def _normalize_order(order: dict) -> dict:
    """订单对象归一化，字段口径与 get_user_orders 保持一致。"""
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
    return {
        "order_id": order.get("id"),
        "order_no": order.get("order_no", ""),
        "total_amount": order.get("total_amount", 0),
        "status": status,
        "status_text": ORDER_STATUS_TEXT.get(status, "未知"),
        "is_paid": bool(order.get("is_paid")),
        "created_at": order.get("created_at", ""),
        "items": items,
    }


def _poll_create_result(idempotency_key: str) -> dict | None:
    """轮询下单结果直到 created/failed；超时返回 None。

    单次轮询的瞬时异常（超时/网络抖动）不中断轮询，计入尝试次数。
    """
    for _ in range(POLL_MAX_ATTEMPTS):
        time.sleep(POLL_INTERVAL_SECONDS)
        try:
            payload = _call(
                "GET", "/order/create/result",
                params={"idempotency_key": idempotency_key},
            )
        except BookstoreAPIError:
            continue
        data = payload.get("data")
        if isinstance(data, dict) and data.get("status") in ("created", "failed"):
            return data
    return None


def create_order(items: list[dict]) -> dict:
    """为当前登录用户创建待支付订单（不支付）。

    Args:
        items: 订单项列表 [{"book_id": int, "quantity": int}]，book_id 必须来自检索结果

    幂等键由本工具生成（uuid）；LLM 重试同一意图不会重复下单——
    后端以 (user_id, idempotency_key) 唯一索引保证重复提交返回同一订单。
    """
    items = _validate_items(items)
    idempotency_key = uuid.uuid4().hex

    payload = _call("POST", "/order/create", json_body={
        "idempotency_key": idempotency_key,
        "items": items,
    })
    data = payload.get("data")
    if not isinstance(data, dict):
        raise OrderAPIError("下单接口返回数据格式异常")

    if data.get("status") == "created":
        # 受理快查命中幂等键（重复提交），直接返回已有订单
        return {
            "ok": True,
            "status": "created",
            "order_id": data.get("order_id"),
            "message": "订单已创建（待支付）",
        }

    result = _poll_create_result(idempotency_key)
    if result is None:
        return {
            "ok": False,
            "status": "pending",
            "idempotency_key": idempotency_key,
            "error": (
                "订单已提交但仍在处理中（确认超时）。请告知用户稍后在订单列表查看结果，"
                "在确认结果前不要重复下单"
            ),
        }
    if result.get("status") == "created":
        return {
            "ok": True,
            "status": "created",
            "order_id": result.get("order_id"),
            "order_no": result.get("order_no", ""),
            "message": "订单已创建（待支付），30 分钟内未支付将自动取消",
        }
    reason = str(result.get("message") or "下单失败")
    return {"ok": False, "status": "failed", "error": reason}


def cancel_order(order_id: int) -> dict:
    """取消当前用户的待支付订单（仅待支付状态可取消，已取消视为成功）

    Args:
        order_id: 订单 ID
    """
    order_id = _validate_order_id(order_id)
    _call("POST", f"/order/{order_id}/cancel")
    return {"ok": True, "order_id": order_id, "message": "订单已取消"}


def get_order_detail(order_id: int) -> dict:
    """获取当前用户单笔订单的详情（含订单内图书、金额与状态）

    Args:
        order_id: 订单 ID
    """
    order_id = _validate_order_id(order_id)
    payload = _call("GET", f"/order/{order_id}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise OrderAPIError("接口返回数据格式异常")
    return {"ok": True, "order": _normalize_order(data)}
