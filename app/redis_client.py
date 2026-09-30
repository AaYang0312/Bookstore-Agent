# Redis 客户端（会话记忆 / 画像缓存 / 入库水位共用）
import logging

from app.config import settings

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:  # 依赖缺失时相关能力自动降级
    redis = None
    REDIS_AVAILABLE = False

_client = None


def get_redis():
    """惰性创建全局 Redis 客户端；未配置 REDIS_URL 或依赖缺失时返回 None（降级）。"""
    global _client
    if not REDIS_AVAILABLE or not settings.redis_url:
        return None
    if _client is None:
        _client = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=3.0,
            socket_connect_timeout=3.0,
        )
    return _client


def reset_client() -> None:
    """测试用：重置单例。"""
    global _client
    _client = None


def ready() -> bool:
    """连通性探测，供 /health 使用。"""
    client = get_redis()
    if client is None:
        return False
    try:
        return bool(client.ping())
    except Exception:
        return False
