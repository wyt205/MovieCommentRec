"""llm-pro 智能助手模块（与 crud 等解耦，便于单独观察/维护）。

组成：
- tools.py   ：用 LangChain @tool 把「查电影/推荐/查影评」封装成模型可调用的工具
- agent.py   ：用 LangChain 装配 ChatOpenAI + 工具 + 会话记忆，构成 function-calling Agent
- api/agent.py：FastAPI 路由，对外暴露 /api/agent/chat

模型待定时（.env 未配置 llm_api_key），后端仍可正常启动，
仅 /api/agent/chat 返回 503 友好提示，不影响电影/影评等其它功能。
"""

from app.ai.agent import chat as chat

__all__ = ["chat"]
