"""进程内验证：转型后的『有护栏自主智能体』关键路径是否真的跑通。"""
import os, sys
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)

from app.ai.agent import chat

CASES = [
    ("刺激的电影", "genre synonym → 动作"),
    ("分别推荐一部评分最高和一部热度最高的电影", "multi-sort 多工具调用"),
    ("你知道《你的名字》吗", "片名碰撞（不是问助手名）"),
    ("讲时间循环的电影", "语义检索路由"),
    ("你是谁", "身份固定自介"),
    ("稍等一下，我还没想好", "defer 不查库"),
]

for q, desc in CASES:
    print("\n" + "=" * 70)
    print(f"【{desc}】\n 用户: {q}")
    try:
        ans = chat(q, session_id="verify")
        print(" 助手:", (ans or "").strip()[:600])
    except Exception as e:  # noqa
        print(" ERR:", type(e).__name__, e)
