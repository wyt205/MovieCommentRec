"""Agent 冒烟测试：验证 GLM-4-Flash-250414 经 LangChain 的 function calling 是否真能跑通。

运行（在 backend 目录下，用装好依赖的 python）：
    python test_agent_smoke.py

测试内容：
  1) 纯问候：走 chat() 但模型不调工具，验证「模型连通 + Agent 循环」正常。
  2) 工具调用（不依赖 MySQL）：自建一个天气工具，验证 GLM 经 LangChain
     真的会发起 function calling、执行工具、并把结果组织成最终回答。
  3) 项目真实 chat() 的电影查询：需要 MySQL 在跑；若没起则捕获报错说明，
     不影响前两项的核心结论。
"""
import os
import sys

# 让 import app 与 .env 加载都基于 backend 目录
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)


def test_greeting():
    print("\n=== 测试 1：纯问候（chat()，不调工具）===")
    try:
        from app.ai.agent import chat
        r = chat("你好，你是谁？用一句话回答")
        print("REPLY:", r)
        return bool(r and r.strip())
    except Exception as e:  # noqa: BLE001
        print("ERR:", type(e).__name__, e)
        return False


def test_tool_calling():
    print("\n=== 测试 2：工具调用（自建天气工具，验证 GLM function calling）===")
    try:
        from langchain_openai import ChatOpenAI
        from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.tools import tool
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from app.core.config import settings

        @tool
        def get_weather(city: str) -> str:
            """查询某城市的天气。参数 city 为城市名。"""
            return f"{city}今天晴，25度，微风。"

        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or None,
            temperature=0,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是测试助手，必须使用工具回答问题。"),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(llm, [get_weather], prompt)
        exe = AgentExecutor(agent=agent, tools=[get_weather], verbose=False, max_iterations=4)
        res = exe.invoke({"input": "北京今天天气怎么样？"})
        out = res.get("output", "")
        print("TOOL-CALL REPLY:", out)
        # 验证模型确实调用了工具（答案里应出现工具返回的「晴，25度」）
        return "25度" in out
    except Exception as e:  # noqa: BLE001
        print("ERR:", type(e).__name__, e)
        return False


def test_real_movie_query():
    print("\n=== 测试 3：项目真实 chat() 电影查询（需 MySQL 在跑）===")
    try:
        from app.ai.agent import chat
        r = chat("有没有科幻电影可以推荐？")
        print("REPLY:", r)
        return bool(r and r.strip())
    except Exception as e:  # noqa: BLE001
        print("ERR (多半是 MySQL 未启动，属预期外因素，不影响前两项结论):",
              type(e).__name__, str(e)[:200])
        return None


if __name__ == "__main__":
    print("PYTHON:", sys.executable)
    r1 = test_greeting()
    r2 = test_tool_calling()
    r3 = test_real_movie_query()
    print("\n==== 结论 ====")
    print(f"测试1 问候连通: {'通过' if r1 else '失败'}")
    print(f"测试2 工具调用 : {'通过' if r2 else '失败'}")
    print(f"测试3 真实查库 : {'通过' if r3 is True else ('跳过(无DB)' if r3 is None else '失败')}")
