# Agent 核心：流式生成器 + 工具循环（同步 /chat 与 SSE /chat/stream 共用同一核心）
import logging

from openai import OpenAI

from app.config import settings
from app.agent.prompt import SYSTEM_PROMPT
from app.agent.tool_prompt import PERSONALIZATION_PROMPT, TOOL_PROMPT
from app.schemas import ChatMessage
from app.tools.registry import TOOLS, execute_tool

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5

# 初始化客户端
client = OpenAI(api_key=settings.model_api_key.get_secret_value(),
                base_url=settings.model_base_url)


def _convert_history(history: list[ChatMessage] | list[dict]) -> list[dict]:
    """转换历史消息为 OpenAI 格式（兼容 ChatMessage 与 dict 输入）"""
    converted = []
    for msg in history:
        if isinstance(msg, ChatMessage):
            converted.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg, dict) and msg.get("role") and msg.get("content"):
            converted.append({"role": msg["role"], "content": msg["content"]})
    return converted


def _build_system_content(profile_segment: str | None) -> str:
    system_content = f"{SYSTEM_PROMPT}\n\n{TOOL_PROMPT}\n\n{PERSONALIZATION_PROMPT}"
    if profile_segment:
        system_content = f"{system_content}\n\n{profile_segment}"
    return system_content


def run_agent(history: list, message: str, profile_segment: str | None = None,
              summary: str | None = None):
    """生成器核心：逐事件产出对话过程。

    事件类型：
      {"type": "delta", "content": str}          文本片段
      {"type": "tool_start", "name": str, "arguments": str}
      {"type": "tool_end", "name": str, "result": str}
      {"type": "final", "content": str}          最终完整回答
      {"type": "error", "message": str}
    """
    messages = [{"role": "system", "content": _build_system_content(profile_segment)}]
    if summary:
        messages.append({"role": "system", "content": f"## 之前对话的摘要\n{summary}"})
    messages.extend(_convert_history(history))
    messages.append({"role": "user", "content": message})

    try:
        for _round in range(MAX_TOOL_ROUNDS + 1):
            # 流式调用：同步接口与 SSE 消费同一事件流，避免两套逻辑漂移
            stream = client.chat.completions.create(
                model=settings.model_name,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                stream=True,
            )

            content_parts: list[str] = []
            tool_calls_acc: dict[int, dict] = {}

            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta is None:
                    continue
                if delta.content:
                    content_parts.append(delta.content)
                    yield {"type": "delta", "content": delta.content}
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        acc = tool_calls_acc.setdefault(
                            tc.index, {"id": "", "name": "", "arguments": ""}
                        )
                        if tc.id:
                            acc["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                acc["name"] += tc.function.name
                            if tc.function.arguments:
                                acc["arguments"] += tc.function.arguments

            assistant_content = "".join(content_parts)

            if not tool_calls_acc:
                # 无工具调用：本轮流式输出即最终回答
                if not assistant_content:
                    yield {"type": "error", "message": "模型返回空响应"}
                    return
                yield {"type": "final", "content": assistant_content}
                return

            if _round == MAX_TOOL_ROUNDS:
                yield {
                    "type": "error",
                    "message": (
                        f"模型工具调用次数过多（已超过 {MAX_TOOL_ROUNDS} 轮），"
                        "无法生成最终回答。"
                    ),
                }
                return

            # 有工具调用：按 index 还原工具调用列表，追加 assistant 消息
            tool_calls_payload = []
            for index in sorted(tool_calls_acc):
                acc = tool_calls_acc[index]
                if not acc["name"]:
                    continue
                tool_calls_payload.append({
                    "id": acc["id"] or f"call_{index}",
                    "type": "function",
                    "function": {"name": acc["name"], "arguments": acc["arguments"]},
                })
            if not tool_calls_payload:
                yield {"type": "error", "message": "模型返回空响应"}
                return

            messages.append({
                "role": "assistant",
                "content": assistant_content or None,
                "tool_calls": tool_calls_payload,
            })

            # 逐个执行工具并回填结果
            for tool_call in tool_calls_payload:
                yield {
                    "type": "tool_start",
                    "name": tool_call["function"]["name"],
                    "arguments": tool_call["function"]["arguments"],
                }
                result = execute_tool(
                    tool_call["function"]["name"],
                    tool_call["function"]["arguments"],
                )
                yield {
                    "type": "tool_end",
                    "name": tool_call["function"]["name"],
                    "result": result,
                }
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result,
                })
            # 继续下一轮，让模型基于工具结果生成回答
    except Exception as e:
        logger.exception("Agent 运行异常")
        yield {"type": "error", "message": str(e)}
        return


def model_call(history: list[ChatMessage], message: str,
               profile_segment: str | None = None,
               summary: str | None = None) -> str:
    """
    调用模型 API 并返回文本（同步聚合 run_agent 事件流的兼容入口）

    Args:
        history: 历史消息列表
        message: 当前用户消息
        profile_segment: 可选的用户画像 System Prompt 段
        summary: 可选的历史摘要段

    Returns:
        模型响应的文本内容
    """
    final_text: str | None = None
    for event in run_agent(history, message, profile_segment, summary):
        if event["type"] == "final":
            final_text = event["content"]
        elif event["type"] == "error":
            raise RuntimeError(event["message"])

    if final_text is None:
        raise RuntimeError("模型未返回最终回答")
    return final_text
