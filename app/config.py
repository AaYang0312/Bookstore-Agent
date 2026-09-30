# 设置
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr


class Settings(BaseSettings):
    # 大模型配置
    model_name: str = Field(min_length=1)
    model_base_url: str | None = None
    model_api_key: SecretStr = Field(min_length=1)

    # 网络配置
    bookstore_api_base_url: str = "http://localhost:8080/api/v1"
    bookstore_admin_base_url: str = "http://localhost:8080/admin"
    tavily_api_key: SecretStr = Field(min_length=1)

    # Milvus 向量数据库（未配置时 RAG 整体降级）
    milvus_uri: str | None = None
    milvus_collection_prefix: str = "books"

    # Embedding 服务（OpenAI 兼容接口；未配置模型名时 RAG 整体降级）
    embedding_model_name: str | None = None
    embedding_base_url: str | None = None  # 默认复用 MODEL_BASE_URL
    embedding_api_key: SecretStr | None = None  # 默认复用 MODEL_API_KEY
    embedding_dim: int = Field(default=1024, ge=1)
    embedding_batch_size: int = Field(default=64, ge=1, le=256)
    embedding_timeout: float = Field(default=30.0, gt=0)

    # RAG 检索参数
    rag_recall_k: int = Field(default=20, ge=1, le=100)
    rag_top_k: int = Field(default=5, ge=1, le=20)

    # Rerank 重排（可选；未启用时向量召回结果直接截断）
    rerank_enabled: bool = False
    rerank_base_url: str | None = None
    rerank_api_key: SecretStr | None = None
    rerank_model: str = ""
    rerank_timeout: float = Field(default=10.0, gt=0)

    # Redis（服务端会话记忆、画像缓存、水位与去重；未配置时相关能力降级）
    redis_url: str | None = None

    # 入库管道
    ingest_page_size: int = Field(default=100, ge=1, le=500)
    ingest_schedule_enabled: bool = False
    ingest_schedule_interval_minutes: int = Field(default=60, ge=1)

    # Go 内部只读同步接口共享密钥（X-Internal-Token）
    internal_sync_secret: SecretStr | None = None

    # 对话记忆
    memory_window: int = Field(default=20, ge=2)
    memory_ttl_seconds: int = Field(default=7 * 24 * 3600, ge=60)
    memory_summarize_threshold: int = Field(default=10, ge=1)
    summary_max_chars: int = Field(default=300, ge=50)

    # 用户画像
    profile_cache_ttl_seconds: int = Field(default=60, ge=1)

    # 读取行为
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",

    )

    @property
    def rag_ready(self) -> bool:
        """RAG 是否可用：Milvus 与 Embedding 模型均已配置。"""
        return bool(self.milvus_uri and self.embedding_model_name)

    @property
    def effective_embedding_base_url(self) -> str | None:
        return self.embedding_base_url or self.model_base_url

    @property
    def effective_embedding_api_key(self) -> SecretStr | None:
        return self.embedding_api_key or self.model_api_key

    @property
    def collection_name(self) -> str:
        """collection 名包含维度，切换模型/维度时自动使用新集合并全量重灌。"""
        return f"{self.milvus_collection_prefix}_{self.embedding_dim}"


settings = Settings()
