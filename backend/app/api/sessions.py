"""会话管理 API：支撑前端「对话记录列表」——列出 / 新建 / 读取 / 删除会话。

会话历史的本体由 agent 的 DBChatMessageHistory 自动写入 chat_sessions 表（每次对话落库），
本模块只负责「管理」这张表：前端左侧的会话列表、点击继续聊、删除重开都走这里。
"""

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models import ChatSession

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("")
def list_sessions(db: Session = Depends(get_db)):
    """列出全部会话（按最近更新倒序），供前端左侧列表渲染。"""
    rows = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
    return [
        {
            "session_id": r.session_id,
            "title": r.title or "新对话",
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            "count": len(r.messages or []),
        }
        for r in rows
    ]


@router.post("")
def create_session(db: Session = Depends(get_db)):
    """新建一个空会话，返回 session_id（前端持有它来收发消息）。"""
    sid = "sess-" + uuid.uuid4().hex[:16]
    row = ChatSession(session_id=sid, title="新对话", messages=[])
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"session_id": row.session_id, "title": row.title}


@router.get("/{session_id}")
def get_session(session_id: str, db: Session = Depends(get_db)):
    """读取某会话的完整消息历史（前端点击列表项时加载）。"""
    row = db.query(ChatSession).filter_by(session_id=session_id).first()
    if not row:
        return {"session_id": session_id, "title": "新对话", "messages": []}
    return {
        "session_id": row.session_id,
        "title": row.title or "新对话",
        "messages": row.messages or [],
    }


@router.delete("/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    """删除会话（前端删除列表项时调用）。"""
    row = db.query(ChatSession).filter_by(session_id=session_id).first()
    if row:
        db.delete(row)
        db.commit()
    return {"ok": True, "session_id": session_id}
