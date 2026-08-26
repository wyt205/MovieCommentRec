"""智能助手路由：对外暴露 /api/agent/chat 与 /api/agent/chat/stream。

采用「懒加载 + 友好降级」策略：
- 运行时才 import app.ai，因此即使 langchain 未安装，后端其它功能仍能正常启动。
- 若未配置模型（llm_api_key 为空）或 langchain 未安装，返回 503 + 清晰提示，
  而不是让整个后端崩掉。
- /chat/stream 用 SSE（Server-Sent Events）逐 token 推送答案，结束再推一条 meta，
  即「大模型流式输出 / 流式开发」的标准实现。
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from app.runtime_flags import is_autonomous, set_autonomous

router = APIRouter(prefix="/agent", tags=["agent"])


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ModeRequest(BaseModel):
    autonomous: bool


@router.get("/mode")
def get_mode():
    """读取当前「自主 Agent 模式」开关状态。"""
    return {"autonomous": is_autonomous()}


@router.post("/mode")
def post_mode(req: ModeRequest):
    """切换「自主 Agent 模式」：true=以 LLM 自主 function calling 为主路径（展示工具选择+护栏）；
    false=默认规则路由主路径（可靠、零幻觉）。"""
    return {"autonomous": set_autonomous(req.autonomous)}


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


@router.post("/chat/stream")
def agent_chat_stream(req: ChatRequest):
    """流式对话（SSE）：逐 token 推送答案，结束再推一条 meta（可观测数据）。

    事件协议：
      data: "<text>"            —— 默认 message 事件，每片文本（前端逐字追加）
      event: done\\ndata: {...}  —— 对话结束 + 元数据（意图/工具链/护栏/缓存/耗时）
      event: error\\ndata: {...} —— 出错
    """
    try:
        from app.ai.agent import chat_stream as run_stream
    except ImportError as e:
        err = json.dumps({"detail": f"Agent 尚未启用：未安装 langchain（{e}）"}, ensure_ascii=False)
        return StreamingResponse(
            (f"event: error\ndata: {err}\n\n",),
            media_type="text/event-stream",
        )

    def event_gen():
        try:
            for item in run_stream(req.message, req.session_id):
                if item["type"] == "token":
                    # 默认 event=message，前端按 message 事件收 token
                    yield f"data: {json.dumps(item['text'], ensure_ascii=False)}\n\n"
                elif item["type"] == "status":
                    # 过程状态（如「正在检索电影库…」），前端立即显示，证明流式管道活着
                    yield f"event: status\ndata: {json.dumps(item['text'], ensure_ascii=False)}\n\n"
                elif item["type"] == "done":
                    yield f"event: done\ndata: {json.dumps(item['meta'], ensure_ascii=False)}\n\n"
        except RuntimeError as e:
            yield f"event: error\ndata: {json.dumps({'detail': f'Agent 尚未启用：{e}'}, ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"event: error\ndata: {json.dumps({'detail': f'{type(e).__name__}: {e}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
