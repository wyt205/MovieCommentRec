"""电影分类（类型）目录接口：返回 TMDb 官方 19 个类型，作为前端分类标签的
单一数据源，保证「分类严格来自 API」。"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app import models

router = APIRouter(prefix="/genres", tags=["genres"])


@router.get("")
def list_genres(db: Session = Depends(get_db)):
    """返回分类目录：[{id: tmdb_id, name: 中文名}, ...]，按 TMDb id 排序。"""
    rows = db.query(models.Genre).order_by(models.Genre.tmdb_id).all()
    return [{"id": g.tmdb_id, "name": g.name} for g in rows]
