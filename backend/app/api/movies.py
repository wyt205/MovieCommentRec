from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import func
from sqlalchemy.orm import Session

import requests
from app.db.database import get_db
from app import crud, models, schemas
from app.crawler.tmdb import _detect_proxies

router = APIRouter(prefix="/movies", tags=["movies"])

# 1x1 透明 PNG，作为无海报时的占位图（避免前端 <img> 报 404）
_PLACEHOLDER_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6360000002000154a24f5d0000000049454e44ae42"
    "6082"
)


@router.get("", response_model=list[schemas.MovieOut])
def list_movies(
    keyword: str | None = None,
    genre: str | None = None,
    year: int | None = None,
    sort: str = "rating",  # rating | year | recent
    skip: int = 0,
    limit: int = Query(20, le=100),
    db: Session = Depends(get_db),
):
    return crud.get_movies(
        db, skip=skip, limit=limit, keyword=keyword, genre=genre, year=year, sort=sort
    )


@router.get("/stats")
def movie_stats(db: Session = Depends(get_db)):
    """站点统计：电影总数 / 影评总数 / 平均评分（用于首页头条）。"""
    movies_count = db.query(func.count(models.Movie.id)).scalar() or 0
    reviews_count = db.query(func.count(models.Review.id)).scalar() or 0
    avg = db.query(func.avg(models.Movie.rating)).scalar()
    return {
        "movies": movies_count,
        "reviews": reviews_count,
        "avg_rating": round(float(avg), 2) if avg else None,
    }


@router.get("/{movie_id}", response_model=schemas.MovieDetailOut)
def movie_detail(movie_id: int, db: Session = Depends(get_db)):
    movie = crud.get_movie(db, movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="movie not found")
    reviews = crud.get_reviews_by_movie(db, movie_id, limit=20)
    cast = crud.get_cast_by_movie(db, movie_id)
    dist = crud.get_rating_distribution(db, movie_id)
    review_count = db.query(func.count(models.Review.id)).filter(
        models.Review.movie_id == movie_id).scalar() or 0
    # 用完整版 MovieDetailOut 直接映射 ORM 对象（含 release_date / country /
    # original_title / tagline 等全部扩展字段），再手动挂上计算字段。
    obj = schemas.MovieDetailOut.model_validate(movie)
    obj.reviews = [schemas.ReviewOut.model_validate(r) for r in reviews]
    obj.cast = [schemas.CastOut.model_validate(c) for c in cast]
    obj.rating_distribution = dist
    obj.review_count = review_count
    return obj


@router.get("/{movie_id}/poster")
def movie_poster(movie_id: int, db: Session = Depends(get_db)):
    """海报接口：从 movies.poster（BLOB）回传图片字节；无图时返回占位图。"""
    movie = crud.get_movie(db, movie_id)
    if movie and movie.poster:
        return Response(content=movie.poster, media_type="image/jpeg")
    return Response(content=_PLACEHOLDER_PNG, media_type="image/png")


@router.get("/cast-photo")
def cast_photo(path: str, db: Session = Depends(get_db)):
    """演员头像代理：image.tmdb.org 在国内常不通，后端经代理按需下载后回传字节，
    避免前端直连被墙。无 path 或下载失败返回占位图。"""
    if not path:
        return Response(content=_PLACEHOLDER_PNG, media_type="image/png")
    url = f"https://image.tmdb.org/t/p/w185/{path.lstrip('/')}"
    try:
        proxies = _detect_proxies()
        r = requests.get(url, timeout=20, proxies=proxies or None)
        if r.status_code == 200 and r.content:
            return Response(content=r.content, media_type="image/jpeg")
    except Exception:  # noqa: BLE001
        pass
    return Response(content=_PLACEHOLDER_PNG, media_type="image/png")
