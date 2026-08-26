"""智能助手路由：对外暴露 /api/agent/chat。

采用「懒加载 + 友好降级」策略：
- 运行时才 import app.ai，因此即使 langchain 未安装，后端其它功能仍能正常启动。
- 若未配置模型（llm_api_key 为空）或 langchain 未安装，返回 503 + 清晰提示，
  而不是让整个后端崩掉。
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/agent", tags=["agent"])


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str


@router.post("/chat", response_model=ChatResponse)
def agent_chat(req: ChatRequest):
    # 懒加载 Agent 模块（langchain 可能未安装）
    try:
        from app.ai.agent import chat as run_agent
    except ImportError as e:
        return ChatResponse(
            reply=f"🤖 Agent 尚未启用：未安装 langchain（pip install -r requirements.txt 后重试）。"
        )

    try:
        reply = run_agent(req.message, req.session_id)
    except RuntimeError as e:
        # 未配置模型密钥
        return ChatResponse(reply=f"🤖 Agent 尚未启用：{e}")
    except Exception as e:  # noqa: BLE001
        return ChatResponse(reply=f"⚠️ Agent 执行出错：{type(e).__name__}: {e}")

    return ChatResponse(reply=reply)
