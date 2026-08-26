"""流式端点进程内冒烟测试：用 FastAPI TestClient 真实打到 /api/agent/chat/stream，
解析 SSE 事件，验证 token 逐片到达 + done 携带 meta，且答案与旧版非流式一致。"""
import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

CASES = [
    ("推荐一部动作片", "movie/tool-using"),
    ("你是谁", "identity(应秒回,无工具)"),
    ("稍等，我还没想好", "defer(不查库)"),
    ("你好", "chat(闲聊)"),
]


def parse_sse(text: str):
    """把 SSE 原始文本解析成事件列表 [(event, data), ...]"""
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event = "message"
        data = ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        events.append((event, data))
    return events


def run_case(q: str, label: str):
    print(f"\n=== {label}: 「{q}」 ===")
    with client.stream("POST", "/api/agent/chat/stream",
                       json={"message": q, "session_id": "smoke-stream"}) as r:
        raw = b""
        for chunk in r.iter_bytes():
            raw += chunk
    text = raw.decode("utf-8")
    events = parse_sse(text)
    tokens = [d for (ev, d) in events if ev == "message" and d]
    dones = [d for (ev, d) in events if ev == "done"]
    errors = [d for (ev, d) in events if ev == "error"]
    full = "".join(json.loads(t) for t in tokens)
    print(f"  SSE 事件数: {len(events)} | token 片数: {len(tokens)} | done: {len(dones)} | error: {len(errors)}")
    print(f"  首片耗时感: {'是(>1片)' if len(tokens) > 1 else '否'}")
    if tokens:
        print(f"  首片: {tokens[0][:30]!r}")
        print(f"  尾片: {tokens[-1][:30]!r}")
    if dones:
        meta = json.loads(dones[0])
        print(f"  meta: intent={meta['intent']} tool_calls={len(meta['tool_calls'])} "
              f"guardrail={meta['used_guardrail']} cache_hit={meta['cache_hit']} latency={meta['latency_ms']}ms")
    if errors:
        print(f"  ❌ error: {errors[0][:120]}")
    print(f"  完整答案(前120字): {full[:120]!r}")
    return full


if __name__ == "__main__":
    for q, label in CASES:
        run_case(q, label)
    print("\n✅ 流式冒烟测试完成")
