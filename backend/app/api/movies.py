from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from sqlalchemy import func
from sqlalchemy.orm import Session
import os
import threading

import requests
from app.db.database import get_db, SessionLocal
from app import crud, models, schemas

# ── 海报落盘：彻底移除「每请求同步读 BLOB + 进程内缓存无限增长」的脆弱链路 ──
# 旧实现：每个 <img> 请求都回源 MySQL 读 LONGBLOB，首页一开 20~190 个并发就把
# uvicorn 线程池 + DB 连接池打满，曾导致全站卡死 / 连接被重置 / 后端被拖垮。
# 新实现：海报 BLOB 仅首次访问时回源一次、落盘到 static/posters/{id}.jpg；
# 之后所有请求由 FileResponse 从磁盘直出——走 ASGI 文件流、不占线程池、
# 不查库、不把 BLOB 读进内存，且重启后磁盘缓存仍在，首屏直接秒出。
_API_DIR = os.path.dirname(os.path.abspath(__file__))      # backend/app/api
_APP_DIR = os.path.dirname(_API_DIR)                       # backend/app
_BACKEND_DIR = os.path.dirname(_APP_DIR)                   # backend
POSTER_DIR = os.path.join(_BACKEND_DIR, "static", "posters")
os.makedirs(POSTER_DIR, exist_ok=True)

# 每部电影一把锁，避免并发首请求重复落盘（双重检查锁）
_poster_locks: dict[int, threading.Lock] = {}
_poster_locks_guard = threading.Lock()


def _poster_path(movie_id: int, ext: str = "jpg") -> str:
    return os.path.join(POSTER_DIR, f"{movie_id}.{ext}")


def _get_poster_lock(movie_id: int) -> threading.Lock:
    with _poster_locks_guard:
        return _poster_locks.setdefault(movie_id, threading.Lock())


def _serve_poster_file(movie_id: int):
    """命中磁盘缓存时直接回文件；jpg / png 都兼容。"""
    for ext, mt in (("jpg", "image/jpeg"), ("png", "image/png")):
        p = _poster_path(movie_id, ext)
        if os.path.exists(p):
            return FileResponse(
                p, media_type=mt,
                headers={"Cache-Control": "public, max-age=86400"},
            )
    return None


def warm_posters():
    """启动后后台把全部海报 BLOB 落盘到 static/posters/，使首页/分类页首屏
    直接从磁盘秒出，彻底不碰 DB、不占线程池。失败静默跳过。"""
    try:
        db = SessionLocal()
        try:
            rows = db.query(models.Movie).filter(
                models.Movie.poster.isnot(None)).all()
            for m in rows:
                try:
                    p = _poster_path(m.id, "jpg")
                    if not os.path.exists(p):
                        with open(p, "wb") as f:
                            f.write(m.poster)
                except Exception:  # noqa: BLE001
                    continue
            print(f"[poster] 预热完成：{len(rows)} 张海报已落盘 → {POSTER_DIR}")
        finally:
            db.close()
    except Exception as e:  # noqa: BLE001
        print(f"[poster] 预热跳过（请确认 MySQL 已启动）：{e}")


# 演员头像经 image.tmdb.org 代理，国内常不通 → 旧实现每次 requests.get 阻塞到
# timeout=20，详情页并发十几个就把 uvicorn 线程池占满，导致全站卡死。
# 改为：直连 + 短超时(6s)，结果（含失败占位）进缓存，只阻塞这一次。
_cast_cache: dict[str, bytes] = {}
_cast_lock = threading.Lock()

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
    """海报接口：优先从磁盘 static/posters/{id}.jpg 直出（FileResponse 走 ASGI
    文件流，不占线程池、不查库、不把 BLOB 读进内存）；仅首次访问才回源 DB 落盘一次。

    这彻底替代了「每请求同步读 BLOB + 进程内缓存无限增长」的旧链路——后者在并发
    切分类/翻页时会扇出大量同步查库请求，曾导致线程池打满、全站卡死、连接被重置。
    """
    hit = _serve_poster_file(movie_id)
    if hit is not None:
        return hit
    # 首次访问：加锁回源一次（双重检查，避免并发重复落盘）
    lock = _get_poster_lock(movie_id)
    with lock:
        hit = _serve_poster_file(movie_id)
        if hit is not None:
            return hit
        movie = crud.get_movie(db, movie_id)
        if movie and movie.poster:
            ext = "png" if movie.poster[:4] == b"\x89PNG" else "jpg"
            p = _poster_path(movie_id, ext)
            with open(p, "wb") as f:
                f.write(movie.poster)
            return FileResponse(
                p, media_type=f"image/{ext}",
                headers={"Cache-Control": "public, max-age=86400"},
            )
        # 无海报：落占位图，避免每次 404 + 重复回源
        p = _poster_path(movie_id, "png")
        with open(p, "wb") as f:
            f.write(_PLACEHOLDER_PNG)
        return FileResponse(
            p, media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )


@router.get("/cast-photo")
def cast_photo(path: str):
    """演员头像代理：image.tmdb.org 在国内常不通，旧实现每次 requests.get 阻塞到
    timeout=20，详情页并发十几个就把 uvicorn 线程池占满 → 全站卡死。
    修复：直连 + 短超时(6s)，结果（含失败占位）进缓存，只阻塞这一次，之后秒回。"""
    if not path:
        return Response(content=_PLACEHOLDER_PNG, media_type="image/png")
    with _cast_lock:
        cached = _cast_cache.get(path)
    if cached is not None:
        return Response(content=cached, media_type="image/jpeg")
    url = f"https://image.tmdb.org/t/p/w185/{path.lstrip('/')}"
    data = _PLACEHOLDER_PNG
    try:
        # 直连、短超时：不通就占位，绝不让线程卡 20s 把池子打满
        r = requests.get(url, timeout=6)
        if r.status_code == 200 and r.content:
            data = r.content
    except Exception:  # noqa: BLE001
        pass
    with _cast_lock:
        _cast_cache[path] = data
    return Response(content=data, media_type="image/jpeg")
