"""管理端 API：为独立 admin/ 前端提供「可观测性」数据接口。

端点（均挂在 /api 前缀下，即 /api/admin/*）：
- POST /admin/chat        —— 带元数据返回的对话（含本次调了哪些工具 / 是否走护栏 / 是否缓存命中 / 耗时）
- GET  /admin/traces       —— 最近对话埋点列表（日志面板）
- GET  /admin/stats         —— 汇总指标（总量 / 护栏使用率 / 缓存命中率 / 平均耗时 / 意图分布）
- GET  /admin/eval          —— 读取最近一次评测结果（run_eval.py 产出）
- POST /admin/eval/run      —— 后台触发 run_eval.py（不阻塞请求，前端轮询 GET /admin/eval）

所有接口只读 MySQL 里的 agent_traces / 评测结果，不修改业务数据。
"""
import json
import os
import subprocess
import threading

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import func, select

from app.ai.agent import chat_with_meta
from app.db.database import SessionLocal
from app.models import AgentTrace

router = APIRouter(prefix="/admin", tags=["admin"])

# 评测状态（进程内）。run_eval.py 在后台线程跑，前端轮询 GET /admin/eval 读取结果。
_EVAL_STATE = {"running": False, "last_started_at": None, "last_error": None}
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EVAL_SCRIPT = os.path.join(_BACKEND_DIR, "run_eval.py")
_EVAL_RESULT = os.path.join(_BACKEND_DIR, "eval_result.json")


# ----------------------------------------------------------------------- 对话（带元数据）
class AdminChatRequest(BaseModel):
    message: str
    session_id: str = "admin"


@router.post("/chat")
def admin_chat(req: AdminChatRequest):
    reply, meta = chat_with_meta(req.message, req.session_id)
    return {"reply": reply, **meta}


# ----------------------------------------------------------------------- 埋点列表（日志）
@router.get("/traces")
def list_traces(limit: int = 100):
    db = SessionLocal()
    try:
        rows = (
            db.execute(
                select(AgentTrace)
                .order_by(AgentTrace.created_at.desc())
                .limit(max(1, min(limit, 500)))
            )
            .scalars()
            .all()
        )
        return {
            "total_shown": len(rows),
            "traces": [
                {
                    "id": t.id,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                    "session_id": t.session_id,
                    "query": t.query,
                    "intent": t.intent,
                    "tool_calls": t.tool_calls or [],
                    "used_guardrail": bool(t.used_guardrail),
                    "cache_hit": bool(t.cache_hit),
                    "latency_ms": t.latency_ms,
                    "answer": (t.answer or "")[:2000],
                }
                for t in rows
            ],
        }
    finally:
        db.close()


# ----------------------------------------------------------------------- 汇总指标
@router.get("/stats")
def stats():
    db = SessionLocal()
    try:
        total = db.scalar(select(func.count()).select_from(AgentTrace)) or 0
        if total == 0:
            return {
                "total": 0, "guardrail_count": 0, "cache_hit_count": 0,
                "guardrail_rate": 0.0, "cache_hit_rate": 0.0,
                "avg_latency_ms": 0, "intent_breakdown": {},
            }
        guardrail = db.scalar(
            select(func.count()).select_from(AgentTrace).where(AgentTrace.used_guardrail == True)  # noqa: E712
        ) or 0
        cache_hit = db.scalar(
            select(func.count()).select_from(AgentTrace).where(AgentTrace.cache_hit == True)  # noqa: E712
        ) or 0
        avg_latency = db.scalar(select(func.avg(AgentTrace.latency_ms))) or 0

        # 意图分布
        intent_rows = db.execute(
            select(AgentTrace.intent, func.count())
            .group_by(AgentTrace.intent)
            .order_by(func.count().desc())
        ).all()
        intent_breakdown = {intent: cnt for intent, cnt in intent_rows}

        return {
            "total": total,
            "guardrail_count": guardrail,
            "cache_hit_count": cache_hit,
            "guardrail_rate": round(guardrail / total * 100, 1),
            "cache_hit_rate": round(cache_hit / total * 100, 1),
            "avg_latency_ms": int(round(avg_latency or 0)),
            "intent_breakdown": intent_breakdown,
        }
    finally:
        db.close()


# ----------------------------------------------------------------------- 评测
@router.get("/eval")
def get_eval():
    if _EVAL_STATE["running"]:
        return {"running": True, "result": None}
    if not os.path.exists(_EVAL_RESULT):
        return {"running": False, "result": None, "exists": False}
    try:
        with open(_EVAL_RESULT, "r", encoding="utf-8") as f:
            result = json.load(f)
        return {"running": False, "exists": True, "result": result}
    except Exception as e:  # noqa: BLE001
        return {"running": False, "exists": True, "result": None, "error": str(e)[:200]}


@router.post("/eval/run")
def run_eval():
    if _EVAL_STATE["running"]:
        return {"started": False, "msg": "评测已在运行中，请稍候轮询结果"}
    if not os.path.exists(_EVAL_SCRIPT):
        return {"started": False, "msg": f"未找到评测脚本：{_EVAL_SCRIPT}"}

    import sys
    backend_python = sys.executable  # 启动器已用干净环境跑后端，这里沿用即可

    def _worker():
        _EVAL_STATE["running"] = True
        _EVAL_STATE["last_error"] = None
        try:
            subprocess.run(
                [backend_python, _EVAL_SCRIPT, "--quiet"],
                cwd=_BACKEND_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=600,
            )
        except Exception as e:  # noqa: BLE001
            _EVAL_STATE["last_error"] = str(e)[:200]
        finally:
            _EVAL_STATE["running"] = False

    threading.Thread(target=_worker, daemon=True).start()
    import datetime
    _EVAL_STATE["last_started_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    return {"started": True, "msg": "已后台启动评测（约 1-3 分钟，前端会自动轮询结果）"}
