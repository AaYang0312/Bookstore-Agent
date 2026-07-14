# 前端请求和返回的数据结构
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str = Field(min_length=1, pattern=r'^(user|assistant)$', description="角色类型")
    content: str = Field(min_length=1, max_length=1500, description="消息内容")

class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1500)
    conversation_id: str
    history: list[ChatMessage] = Field(default_factory=list, max_length=10)

class AgentChatResponse(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    conversation_id: str