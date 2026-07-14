from openai import OpenAI
from app.schemas import ChatMessage
from app.config import settings
from app.agent.prompt import SYSTEM_PROMPT

# 初始化客户端
client = OpenAI(api_key=settings.model_api_key.get_secret_value(),
                base_url=settings.model_base_url)

def _convert_history(history:list[ChatMessage]) -> list[dict]:
    # 转换历史消息为 Openai 格式
    return [{"role": msg.role, "content": msg.content}for msg in history]

def model_call(history: list[ChatMessage], message: str) -> str:
    """
    调用模型 API 并返回文本

    Args:
        history: 历史消息列表
        message: 当前用户消息

    Returns:
        模型响应的文本内容
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *_convert_history(history),
                {"role": "user", "content": message}]

    response = client.chat.completions.create(
        model=settings.model_name,
        messages=messages,
    )

    if not response.choices:
        raise ValueError("模型返回空响应")

    return response.choices[0].message.content or ""
