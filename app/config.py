# 设置
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr


class Settings(BaseSettings):
    # 大模型配置
    model_name: str = Field(min_length=1)
    model_base_url: str | None = None
    model_api_key: SecretStr = Field(min_length=1)

    # 读取行为
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",

    )

settings = Settings()