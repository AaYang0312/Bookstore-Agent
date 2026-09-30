# 前端请求和返回的数据结构
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    # 输入仍只接受 user/assistant；工具消息留痕由服务端记忆（app/memory）直接以
    # dict 存储，不经此模型校验，保持前端输入契约完全不变
    role: str = Field(min_length=1, pattern=r'^(user|assistant)$', description="角色类型")
    content: str = Field(min_length=1, max_length=1500, description="消息内容")

class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1500)
    conversation_id: str
    history: list[ChatMessage] = Field(default_factory=list, max_length=10)

class AgentChatResponse(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    conversation_id: str

# ============ SSE 事件模型（/chat/stream） ============

class DeltaEvent(BaseModel):
    content: str = Field(description="文本片段")

class ToolStartEvent(BaseModel):
    name: str = Field(description="工具名称")
    arguments: dict = Field(default_factory=dict, description="工具参数")

class ToolEndEvent(BaseModel):
    name: str = Field(description="工具名称")
    result: str = Field(default="", description="工具结果摘要（截断）")

class DoneEvent(BaseModel):
    message: str = Field(description="最终完整回答")
    conversation_id: str = Field(description="会话 ID")

class ErrorEvent(BaseModel):
    message: str = Field(description="错误说明")
