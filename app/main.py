# FAST API 路由
from fastapi import FastAPI
from app.schemas import AgentChatRequest, AgentChatResponse
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.agent.harness import model_call

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/v1/agent/chat", response_model=AgentChatResponse)
def chat(request: AgentChatRequest):
    try:
        reply = model_call(history=request.history, message=request.message)
    except RuntimeError as e:
        reply = f"抱歉，处理您的请求时出现问题：{e}"

    return AgentChatResponse(
        message=reply,
        conversation_id=request.conversation_id,
    )

@app.get("/model-info")
def get_model_info():
    return {
        "model": settings.model_name,
    }

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["POST","OPTIONS"],
    allow_headers=["Content-Type","Accept","Authorization"]
)