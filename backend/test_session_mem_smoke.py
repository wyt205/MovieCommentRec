"""进程内冒烟：会话持久化 + 列表/读取/删除 API + 长期偏好抽取落库。"""
import sys
sys.path.insert(0, r"d:\Study\AI大模型相关\Agent\llm-pro\backend")

from app.db.database import Base, engine, SessionLocal
from app.models import ChatSession, UserPreference
Base.metadata.create_all(bind=engine)  # 确保 chat_sessions / user_preferences 已建
# 清理历史残留，保证可重复运行
_db = SessionLocal()
_db.query(ChatSession).delete()
_db.query(UserPreference).delete()
_db.commit()
_db.close()

from fastapi.testclient import TestClient
from app.main import app
from app.models import UserPreference

with TestClient(app) as client:
    # 1) 新建会话
    r = client.post("/api/sessions")
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    print("1) 新建会话 ->", sid)

    # 2) 两轮真实对话（落库到该会话）
    for q in ["推荐一部动作片", "类似《星际穿越》的电影"]:
        rr = client.post("/api/agent/chat", json={"message": q, "session_id": sid})
        assert rr.status_code == 200, rr.text
        print(f"   对话: {q!r} -> {rr.json()['reply'][:36]!r}…")

    # 3) 列表 + 条数
    lst = client.get("/api/sessions").json()
    cnt = [x for x in lst if x["session_id"] == sid][0]["count"]
    print(f"3) 会话列表条数={len(lst)}，本会话消息数={cnt} (期望 4)")

    # 4) 读取历史
    msgs = client.get(f"/api/sessions/{sid}").json()["messages"]
    print(f"4) 读取历史消息数={len(msgs)}，roles={[m['role'] for m in msgs]}")

    # 5) 长期偏好抽取
    rr = client.post("/api/agent/chat", json={"message": "我喜欢科幻电影", "session_id": sid})
    assert rr.status_code == 200, rr.text
    db = SessionLocal()
    prefs = [(p.key, p.value) for p in db.query(UserPreference).all()]
    db.close()
    print(f"5) 偏好输入回复={rr.json()['reply'][:30]!r}；落库偏好={prefs}")

    # 6) 删除会话
    rr = client.delete(f"/api/sessions/{sid}")
    after = client.get("/api/sessions").json()
    print(f"6) 删除={rr.json()['ok']}；删除后列表是否仍含本会话={any(x['session_id']==sid for x in after)} (期望 False)")

print("DONE")
