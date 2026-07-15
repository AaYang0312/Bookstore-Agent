from openai import OpenAI
from app.schemas import ChatMessage
from app.config import settings
from app.agent.prompt import SYSTEM_PROMPT
from app.tools.registry import TOOLS, execute_tool

MAX_TOOL_ROUNDS = 5

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
        tools=TOOLS,
        tool_choice="auto"
    )

    if not response.choices:
        raise ValueError("模型返回空响应")

    assistant_message = response.choices[0].message
    for _ in range(MAX_TOOL_ROUNDS):
        if not assistant_message.tool_calls:
            if not assistant_message.content:
                raise ValueError("模型返回空响应")
            return assistant_message.content

        # 有工具调用：执行工具，把结果追加到消息列表
        messages.append(assistant_message.model_dump(exclude_none=True))
        for tool_call in assistant_message.tool_calls:
            result = execute_tool(tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        # 再次调用模型，让它根据工具结果生成回答
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        if not response.choices:
            raise ValueError("模型返回空响应")
        assistant_message = response.choices[0].message

    # 超过最大轮次，模型仍在请求工具调用，无法得出最终回答
    raise RuntimeError(
        f"模型工具调用次数过多（已超过 {MAX_TOOL_ROUNDS} 轮），无法生成最终回答。"
    )
