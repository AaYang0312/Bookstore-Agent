# 超窗历史 LLM 摘要压缩：两级记忆（摘要 + 滑窗），摘要复用避免每轮重复调用
import logging

from openai import OpenAI

from app.config import settings
from app.memory import store

logger = logging.getLogger(__name__)

_client: OpenAI | None = None

SUMMARIZE_PROMPT = """你是图书购物助手的对话摘要器。请把【既有摘要】与【新增对话】压缩为一段不超过 {max_chars} 字的中文摘要，供后续对话作为长期记忆。

必须保留的信息（如有）：
- 用户的阅读偏好、兴趣方向、预算等画像信息
- 已经推荐过或讨论过的书名（避免重复推荐）
- 用户提出但尚未解决的问题

输出只包含摘要正文，不要任何前缀或解释。

【既有摘要】
{previous_summary}

【新增对话】
{dialog}"""


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.model_api_key.get_secret_value(),
            base_url=settings.model_base_url,
        )
    return _client


def summarize_messages(previous_summary: str | None, messages: list[dict]) -> str | None:
    """调用 LLM 生成压缩摘要；失败返回 None（溢出缓冲保留，下次重试）。"""
    dialog_lines = []
    for message in messages:
        role = "用户" if message.get("role") == "user" else "助手"
        dialog_lines.append(f"{role}：{message.get('content', '')}")
    prompt = SUMMARIZE_PROMPT.format(
        max_chars=settings.summary_max_chars,
        previous_summary=previous_summary or "（无）",
        dialog="\n".join(dialog_lines),
    )
    try:
        response = _get_client().chat.completions.create(
            model=settings.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            return None
        return content.strip()
    except Exception as e:
        logger.warning("生成会话摘要失败（下次写回时重试）：%s", e)
        return None


def maybe_compress(conversation_id: str) -> bool:
    """溢出缓冲达到阈值时压缩为摘要；成功返回 True。失败时数据保留在缓冲中。"""
    overflow = store.get_overflow(conversation_id)
    if len(overflow) < settings.memory_summarize_threshold:
        return False

    context = store.load_context(conversation_id)
    summary = summarize_messages(context.get("summary"), overflow)
    if summary is None:
        return False
    return store.save_summary(conversation_id, summary)
