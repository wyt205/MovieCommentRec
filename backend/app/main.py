from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import movies, reviews, genres, agent, admin
from app.db.database import Base, engine, SessionLocal

from app import models  # 确保模型被加载，create_all 才能建表；同时把 models 名绑进作用域供播种使用
from app.core import genres as genres_catalog

# 静态资源目录（海报 / 头像等）：backend/static → 通过 /static 访问
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 开发期自动建表（生产建议用 sql/init.sql + 迁移工具）
    Base.metadata.create_all(bind=engine)
    # 幂等播种分类目录（TMDb 官方 19 个类型）
    _seed_genres()
    yield


def _seed_genres():
    """若 genres 表为空，用 app/core/genres.py 的官方列表填充（幂等）。"""
    try:
        db = SessionLocal()
        try:
            if db.query(models.Genre).count() == 0:
                for g in genres_catalog.TMDB_GENRES:
                    db.add(models.Genre(tmdb_id=g["id"], name=g["name"]))
                db.commit()
                print(f"[seed] 已写入 {len(genres_catalog.TMDB_GENRES)} 个电影分类")
        finally:
            db.close()
    except Exception as e:  # noqa: BLE001
        print(f"[seed] 分类目录播种跳过（请确认 MySQL 已启动）：{e}")


app = FastAPI(title="llm-pro 影评 API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # 开发期：把真实异常回传给前端，便于定位（生产可改为只返回 500）
    import traceback

    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
    )

# 允许前端（Vue 开发服务器）跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(movies.router, prefix="/api")
app.include_router(reviews.router, prefix="/api")
app.include_router(genres.router, prefix="/api")
app.include_router(agent.router, prefix="/api")
app.include_router(admin.router, prefix="/api")

# 托管本地图片（海报等）：浏览器访问 /static/posters/1292052.svg
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    return {"msg": "llm-pro API running", "docs": "/docs"}
