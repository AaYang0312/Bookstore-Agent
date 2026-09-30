# Embedding 客户端（OpenAI 兼容接口，批量 + 指数退避重试 + 维度自检）
import logging
import math
import time

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # 秒，指数退避基数


class EmbeddingError(Exception):
    """Embedding 服务调用异常"""
    pass


class Embedder:
    """OpenAI 兼容 embeddings 客户端。

    向量统一做 L2 归一化，与 Milvus 的 IP 度量配套使用，
    使内积等价于余弦相似度。
    """

    def __init__(self):
        self._client: OpenAI | None = None
        self._observed_dim: int | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            base_url = settings.effective_embedding_base_url
            api_key = settings.effective_embedding_api_key
            if not base_url or api_key is None:
                raise EmbeddingError("Embedding 服务未配置 base_url 或 api_key")
            self._client = OpenAI(
                api_key=api_key.get_secret_value(),
                base_url=base_url,
                timeout=settings.embedding_timeout,
            )
        return self._client

    def _check_dim(self, dim: int) -> None:
        """维度自检：与配置不一致时直接失败，提示切换 collection。"""
        if self._observed_dim is None:
            self._observed_dim = dim
        if dim != settings.embedding_dim:
            raise EmbeddingError(
                f"Embedding 模型实际维度 {dim} 与配置 EMBEDDING_DIM="
                f"{settings.embedding_dim} 不一致，请修正配置（切换维度需全量重灌向量库）"
            )

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm <= 0:
            return vec
        return [x / norm for x in vec]

    def _embed_batch_raw(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._get_client().embeddings.create(
                    model=settings.embedding_model_name,
                    input=texts,
                )
                if not response.data:
                    raise EmbeddingError("Embedding 接口返回空数据")
                vectors = [item.embedding for item in response.data]
                self._check_dim(len(vectors[0]))
                return vectors
            except EmbeddingError:
                raise
            except Exception as e:  # 网络/限流/服务端错误均重试
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Embedding 请求失败（第 %d 次）：%s，%.1fs 后重试",
                        attempt, e, delay,
                    )
                    time.sleep(delay)
        raise EmbeddingError(f"Embedding 请求失败（已重试 {MAX_RETRIES} 次）：{last_error}")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文本，按配置分批，返回归一化后的向量。"""
        if not texts:
            return []
        vectors: list[list[float]] = []
        batch_size = settings.embedding_batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            for vec in self._embed_batch_raw(batch):
                vectors.append(self._normalize(vec))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """单条查询向量化（同样归一化）。"""
        vectors = self.embed_texts([text])
        if not vectors:
            raise EmbeddingError("查询向量化失败")
        return vectors[0]


# 模块级单例，与 harness 中 OpenAI 客户端的用法保持一致
embedder = Embedder()
