# 用户画像：拉取（带 Redis TTL 缓存）→ 渲染进 System Prompt
import hashlib
import json
import logging

import httpx

from app.auth_context import get_token
from app.config import settings
from app.redis_client import get_redis

logger = logging.getLogger(__name__)

PROFILE_CACHE_KEY_PREFIX = "agent:profile:v1"


class ProfileError(Exception):
    """画像拉取异常"""
    pass


def _cache_key(token: str) -> str:
    """以 token 摘要作缓存键，避免原始 token 进入 Redis。"""
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"{PROFILE_CACHE_KEY_PREFIX}:{digest}"


def _create_http_client(timeout: float, trust_env: bool) -> httpx.Client:
    return httpx.Client(timeout=timeout, trust_env=trust_env)


def fetch_profile(token: str | None = None) -> dict | None:
    """拉取聚合画像；未登录/无数据/服务异常时返回 None（不阻塞对话）。"""
    token = token or get_token()
    if not token:
        return None

    client = get_redis()
    if client is not None:
        try:
            cached = client.get(_cache_key(token))
            if cached:
                profile = json.loads(cached)
                if isinstance(profile, dict):
                    return profile
        except Exception:
            pass  # 缓存读取失败不阻塞

    base_url = settings.bookstore_api_base_url.rstrip("/")
    try:
        with _create_http_client(timeout=5.0, trust_env=False) as http:
            response = http.get(
                base_url + "/user/agent-profile",
                headers={"Authorization": f"Bearer {token}"},
            )
        if response.status_code == 401:
            return None  # 未登录/过期：无画像，走通用推荐
        response.raise_for_status()
        payload = response.json()
    except httpx.TimeoutException:
        logger.warning("画像接口请求超时，跳过画像注入")
        return None
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError):
        logger.warning("画像接口不可用，跳过画像注入")
        return None

    if not isinstance(payload, dict) or payload.get("code") != 0:
        return None
    profile = payload.get("data")
    if not isinstance(profile, dict):
        return None

    if client is not None:
        try:
            client.set(_cache_key(token), json.dumps(profile, ensure_ascii=False),
                       ex=settings.profile_cache_ttl_seconds)
        except Exception:
            pass
    return profile


def _render_categories(categories: list) -> str:
    if not categories:
        return ""
    parts = []
    for item in categories:
        if isinstance(item, dict) and item.get("category"):
            parts.append(f"{item['category']}×{item.get('count', 1)}")
    return "、".join(parts[:5])


def render_profile_segment(profile: dict) -> str | None:
    """将画像渲染为 System Prompt 画像段；空画像返回 None。"""
    recent_orders = profile.get("recent_orders") or []
    favorite_categories = profile.get("favorite_categories") or []
    browse_top_categories = profile.get("browse_top_categories") or []
    recent_browsed = profile.get("recent_browsed") or []

    if not (recent_orders or favorite_categories or browse_top_categories or recent_browsed):
        return None

    lines = ["## 用户画像（数据来自书城，用于个性化推荐）"]

    if recent_orders:
        order_parts = []
        for order in recent_orders[:10]:
            if not isinstance(order, dict):
                continue
            titles = [
                str(item.get("title", ""))
                for item in (order.get("items") or [])
                if isinstance(item, dict) and item.get("title")
            ]
            order_parts.append(
                f"订单{order.get('order_id')}（{order.get('status_text', '')}，"
                f"{order.get('total_amount', 0)}元）：{'、'.join(titles[:3])}"
            )
        if order_parts:
            lines.append("- 近期订单：" + "；".join(order_parts[:5]))

    favorite_text = _render_categories(favorite_categories)
    if favorite_text:
        lines.append(f"- 常收藏分类：{favorite_text}")

    browse_text = _render_categories(browse_top_categories)
    if browse_text:
        lines.append(f"- 最近浏览最多的分类：{browse_text}")

    if recent_browsed:
        browsed_parts = []
        for book in recent_browsed[:10]:
            if isinstance(book, dict) and book.get("title"):
                browsed_parts.append(str(book["title"]))
        if browsed_parts:
            lines.append("- 最近在看：" + "、".join(browsed_parts[:10]))

    if len(lines) <= 1:
        return None
    return "\n".join(lines)


def build_profile_segment(token: str | None = None) -> str | None:
    """组合入口：拉取 + 渲染；任何失败均返回 None（画像冷启动不阻塞对话）。"""
    try:
        profile = fetch_profile(token)
        if profile is None:
            return None
        return render_profile_segment(profile)
    except Exception as e:
        logger.warning("构建画像段失败：%s", e)
        return None
