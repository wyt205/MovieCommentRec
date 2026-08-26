from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import SessionLocal, get_db
from app import crud, models, schemas

router = APIRouter(prefix="/reviews", tags=["reviews"])


def _with_movie_title(db: Session, rv) -> schemas.ReviewOut:
    """把 ORM 影评转成 ReviewOut，并补上电影名 / 作者名（供影评广场/详情展示）。"""
    item = schemas.ReviewOut.model_validate(rv)
    movie = db.get(models.Movie, rv.movie_id)
    item.movie_title = movie.title if movie else None
    reviewer = db.get(models.Reviewer, rv.reviewer_id) if rv.reviewer_id else None
    item.reviewer_name = reviewer.name if reviewer else None
    return item


@router.get("", response_model=list[schemas.ReviewOut])
def list_reviews(
    movie_id: int | None = None,
    skip: int = 0,
    limit: int = Query(20, le=100),
    db: Session = Depends(get_db),
):
    q = db.query(models.Review)
    if movie_id:
        q = q.filter(models.Review.movie_id == movie_id)
    rows = (
        q.order_by(models.Review.publish_date.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [_with_movie_title(db, rv) for rv in rows]


@router.get("/{review_id}", response_model=schemas.ReviewOut)
def review_detail(review_id: int, db: Session = Depends(get_db)):
    obj = db.get(models.Review, review_id)
    if not obj:
        raise HTTPException(status_code=404, detail="review not found")
    return _with_movie_title(db, obj)


@router.post("", response_model=schemas.ReviewOut, status_code=201)
def create_review(payload: schemas.ReviewCreate, db: Session = Depends(get_db)):
    """用户在前端点星打分 + 写短评，入库为 source='user' 的评论（轻量用户体系，无需登录）。"""
    movie = db.get(models.Movie, payload.movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="movie not found")
    rv = crud.create_user_review(
        db, payload.movie_id, payload.nickname, payload.rating, payload.content
    )
    return _with_movie_title(db, rv)
