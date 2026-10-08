# Agent 核心：流式生成器 + 工具循环（同步 /chat 与 SSE /chat/stream 共用同一核心）
import json
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


def _base_messages(history: list, profile_segment: str | None,
                   summary: str | None) -> list[dict]:
    """公共消息前缀：System Prompt +（可选摘要）+ 历史。"""
    messages = [{"role": "system", "content": _build_system_content(profile_segment)}]
    if summary:
        messages.append({"role": "system", "content": f"## 之前对话的摘要\n{summary}"})
    messages.extend(_convert_history(history))
    return messages


def _reply_loop(messages: list[dict]):
    """流式回复主循环：LLM 流式输出 + 工具调用回填，最多 MAX_TOOL_ROUNDS 轮。

    事件类型：
      {"type": "delta", "content": str}          文本片段
      {"type": "tool_start", "name": str, "arguments": str}
      {"type": "tool_end", "name": str, "result": str}
      {"type": "confirm_request", "operation_id": str, "summary": dict}
      {"type": "final", "content": str}          最终完整回答
      {"type": "error", "message": str}
    """
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
                # 写操作提议：把确认卡片推给前端（SSE confirm_request 事件）
                confirm = _confirm_event(result)
                if confirm is not None:
                    yield confirm
            # 继续下一轮，让模型基于工具结果生成回答
    except Exception as e:
        logger.exception("Agent 运行异常")
        yield {"type": "error", "message": str(e)}
        return


def _confirm_event(tool_result: str) -> dict | None:
    """propose 工具的返回携带确认卡片时转为 confirm_request 事件，否则返回 None。"""
    try:
        parsed = json.loads(tool_result)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict) and parsed.get("action") == "confirm_required":
        return {
            "type": "confirm_request",
            "operation_id": parsed.get("operation_id"),
            "summary": parsed.get("summary"),
        }
    return None


def run_agent(history: list, message: str, profile_segment: str | None = None,
              summary: str | None = None):
    """生成器核心：逐事件产出对话过程（事件类型见 _reply_loop）。"""
    messages = _base_messages(history, profile_segment, summary)
    messages.append({"role": "user", "content": message})
    yield from _reply_loop(messages)


def run_confirmed(history: list, operation: dict, profile_segment: str | None = None,
                  summary: str | None = None):
    """用户确认后的执行流：用存储的原始参数直接执行写操作，再让 LLM 基于真实结果作答。

    执行不经 LLM——模型物理上无法触发或篡改写操作；LLM 只负责把执行结果组织成回复。
    """
    tool = operation["tool"]
    arguments = json.dumps(operation["arguments"], ensure_ascii=False)
    call_id = f"confirm_{operation['op_id']}"

    result = execute_tool(tool, arguments)

    messages = _base_messages(history, profile_segment, summary)
    messages.append({"role": "user", "content": "（用户已点击确认卡片，同意执行刚才提议的操作）"})
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": tool, "arguments": arguments},
        }],
    })
    messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
    yield from _reply_loop(messages)


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


def model_call_confirmed(history: list, operation: dict,
                         profile_segment: str | None = None,
                         summary: str | None = None) -> str:
    """同步聚合 run_confirmed 事件流的兼容入口（确认执行后的回答）。"""
    final_text: str | None = None
    for event in run_confirmed(history, operation, profile_segment, summary):
        if event["type"] == "final":
            final_text = event["content"]
        elif event["type"] == "error":
            raise RuntimeError(event["message"])

    if final_text is None:
        raise RuntimeError("模型未返回最终回答")
    return final_text
