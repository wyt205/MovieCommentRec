import sys
sys.path.insert(0, r"d:\Study\AI大模型相关\Agent\llm-pro\backend")
import app.ai.agent as A

# 临时打桩：记录模型自主调了哪些工具（验证 autonomous agent 行为）。
# 注意：必须保存并调用【原函数】，否则会无限递归。
_calls = []
_orig = A._safe_invoke_tool

def _patched(tc):
    _calls.append((tc.get("name"), tc.get("args")))
    return _orig(tc)

A._safe_invoke_tool = _patched

cases = [
    ("你知道超能陆战队吗", "t_know"),
    ("推荐五个冒险题材动画", "t_adv"),
    ("分别推一个评分最高和一个热度最高的电影", "t_both"),
    ("你知不知你的名字", "t_name"),
    ("你好", "t_hi"),
    ("你叫什么名字", "t_who"),
    ("我一会儿会给你要求", "t_defer"),
]

for q, sid in cases:
    _calls.clear()
    try:
        r = A.chat(q, sid)
    except Exception as e:
        r = "!!ERR: " + repr(e)
    print("问:", q)
    print("  模型自主调用工具:", _calls if _calls else "(无 / 走确定性分支)")
    print("  答:", r[:200])
    print("-" * 60)
