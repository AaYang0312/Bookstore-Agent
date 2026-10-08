# Go 书城后端 HTTP 访问共享层：JWT 透传 + 统一错误转换（保留非 2xx 响应中的业务 message）
# 供写操作工具（favorite / order）复用；user.py 的只读工具沿用各自实现，行为不变
import logging

import httpx

from app.auth_context import get_token
from app.config import settings

logger = logging.getLogger(__name__)

NOT_LOGGED_IN_HINT = (
    "用户未登录（缺少有效的 Authorization token），无法执行该操作，"
    "请告知用户需要登录后再试，并继续当前话题"
)


class BookstoreAPIError(Exception):
    """书城接口调用异常（message 面向 LLM，应转述给用户，不得重试写操作）"""
    pass


def _create_http_client(timeout: float, trust_env: bool) -> httpx.Client:
    return httpx.Client(timeout=timeout, trust_env=trust_env)


def request_bookstore(method: str, path: str, params: dict | None = None,
                      json_body: dict | None = None,
                      error_class: type[BookstoreAPIError] = BookstoreAPIError) -> dict:
    """携带 JWT 调用书城接口，返回 code==0 的完整 payload。

    非 2xx / code!=0 时抛 error_class（默认基类），message 优先取响应体里的业务提示
    （如“仅待支付订单可以取消”），让 LLM 能向用户转述真实原因。
    """
    token = get_token()
    if not token:
        raise error_class(NOT_LOGGED_IN_HINT)

    base_url = settings.bookstore_api_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with _create_http_client(timeout=5.0, trust_env=False) as client:
            response = client.request(
                method, base_url + path, params=params, json=json_body, headers=headers,
            )
    except httpx.TimeoutException:
        raise error_class("书城接口请求超时，请稍后重试")
    except httpx.RequestError:
        raise error_class("无法连接书城服务，请稍后重试")

    message = ""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        message = str(payload.get("message") or "")

    if response.status_code == 401:
        raise error_class("用户登录状态已失效，请告知用户重新登录后重试")
    if response.status_code == 503:
        # Go 侧 ErrMQUnavailable：消息队列不可用
        raise error_class(message or "书城服务暂时不可用，请稍后重试")
    if not isinstance(payload, dict):
        raise error_class(f"书城接口返回异常（HTTP {response.status_code}）")
    if response.status_code >= 400:
        raise error_class(message or f"书城接口返回错误（HTTP {response.status_code}）")
    if payload.get("code") != 0:
        raise error_class(message or "书城接口调用失败")
    return payload
