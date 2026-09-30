# 请求级 JWT 上下文：FastAPI 中间件提取 Authorization，工具层透传给 Go 后端
# token 仅在请求生命周期内传递，不落盘、不打日志
import contextvars

_token_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "bookstore_auth_token", default=None
)


def set_token(token: str | None) -> None:
    _token_var.set(token)


def get_token() -> str | None:
    return _token_var.get()
