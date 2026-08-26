"""管理端埋点层 + API 的进程内冒烟验证（不依赖端口，直接 import 模块调）。"""
import sys
sys.path.insert(0, ".")

from app.db.database import Base, engine
from app import models  # 加载 AgentTrace
from app.ai.agent import chat_with_meta
from app.api import admin
from app.api.admin import AdminChatRequest

# 1) 确保 agent_traces 表已建（兼容旧库）
Base.metadata.create_all(bind=engine)
print("[1] 建表完成（含 agent_traces）")

# 2) 跑几条对话，验证 meta + 落库
print("\n[2] 对话（含缓存命中验证）")
q1 = "你好"
a1, m1 = chat_with_meta(q1, "smoke")
print(f"  Q1={q1!r} -> intent={m1['intent']} guardrail={m1['used_guardrail']} cache={m1['cache_hit']} {m1['latency_ms']}ms")

q2 = "推荐一部动作片"
a2, m2 = chat_with_meta(q2, "smoke")
print(f"  Q2={q2!r} -> intent={m2['intent']} tools={[t['name'] for t in m2['tool_calls']]} cache={m2['cache_hit']} {m2['latency_ms']}ms")

# 同问题复问 -> 应命中缓存（不调 LLM，cache_hit=True）
a2b, m2b = chat_with_meta(q2, "smoke")
print(f"  Q2(复问) -> cache_hit={m2b['cache_hit']} (期望 True)  latency={m2b['latency_ms']}ms")

# 3) 管理端 API：对话（带元数据）
print("\n[3] 管理端 /admin/chat")
r = admin.admin_chat(AdminChatRequest(message="讲时间循环的电影", session_id="smoke"))
print(f"  reply[:40]={r['reply'][:40]!r}  intent={r['intent']} tools={[t['name'] for t in (r['tool_calls'] or [])]}")

# 4) 日志列表 + 统计
print("\n[4] /admin/traces & /admin/stats")
traces = admin.list_traces(limit=10)
print(f"  traces.total_shown={traces['total_shown']}")
s = admin.stats()
print(f"  stats.total={s['total']} guardrail_rate={s['guardrail_rate']}% cache_hit_rate={s['cache_hit_rate']}% avg={s['avg_latency_ms']}ms")
print(f"  intent_breakdown={s['intent_breakdown']}")

# 5) 评测结果读取（若存在）
print("\n[5] /admin/eval")
e = admin.get_eval()
print(f"  running={e['running']} exists={e.get('exists')} has_result={e.get('result') is not None}")

print("\n✅ 冒烟验证完成")
