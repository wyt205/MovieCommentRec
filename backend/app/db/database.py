from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session, sessionmaker, declarative_base

from app.core.config import settings

# 连接 MySQL（PyMySQL 驱动）
# 关键坑（本项目已踩）：pool_pre_ping=True 只能识别「完全断掉」的连接；
# 若复用到的连接是「半开死连接」（MySQL 重启 / 网络抖动导致客户端没收到 RST），
# pre_ping 的 SELECT 1 探测会【无限挂起】——因为 pymysql 默认无任何 socket 超时。
# 注：本机实测 MySQL wait_timeout=28800秒(8小时)，不会主动掐连接，故“陈旧连接”主因是
# 重启/网络抖动而非 wait_timeout；但 socket 超时仍是通用兜底，保留无害。
# 结果是：死连接被当成存活复用 → 查询永远不返回 → 线程池被逐渐占满 →
# 连不碰库的 /health 都排不上队，前端所有取数「一直加载、像断网」。
# 修复三件套：
#   1) connect_args 里的 read/write/connect_timeout：让探测/查询最多等 30s 就主动报错
#      （而非永挂），pre_ping 据此丢弃死连接、换一条新连接；
#   2) pool_recycle=3600：在 MySQL wait_timeout 之前主动回收连接，从源头避免陈旧连接；
#   3) pool_pre_ping=True 保留：借出连接前先 SELECT 1 兜底探测。
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    connect_args={
        "connect_timeout": 5,   # 建立连接最多等 5s
        "read_timeout": 30,     # 单次查询/探测最多等 30s（超过即视为死连接，报错而非永挂）
        "write_timeout": 30,    # 写入最多等 30s
    },
)

# 会话工厂
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# 所有 ORM 模型的基类
Base = declarative_base()


def get_db():
    """FastAPI 依赖：每次请求一个数据库会话（自动关闭）。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _ensure_database():
    """若 llm_pro 库不存在则自动创建（避免「连不上」只因库没建）。
    需 MySQL 服务已启动且账号密码正确；否则静默跳过，由启动器/用户排查。"""
    try:
        u = make_url(settings.database_url)
        no_db = u.set(database="")
        eng = create_engine(no_db, pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text(
                f"CREATE DATABASE IF NOT EXISTS `{u.database}` "
                "DEFAULT CHARACTER SET utf8mb4 DEFAULT COLLATE utf8mb4_unicode_ci"
            ))
            c.commit()
        eng.dispose()
    except Exception as e:  # noqa: BLE001
        print(f"[db] 数据库自动创建跳过（请确认 MySQL 已启动且账号密码正确）：{e}")


def _migrate_columns():
    """兼容已有库：补齐 TMDb 相关列、删除豆瓣专属列与短评表。
    幂等：仅在列/表缺失或存在豆瓣残留时执行；删除前先判断是否存在；
    无 MySQL 连接时静默跳过（不影响导入）。"""
    try:
        insp = inspect(engine)
        tables = set(insp.get_table_names())
        mcols = {c["name"]: c for c in insp.get_columns("movies")}
        rcols = {c["name"]: c for c in insp.get_columns("reviews")}
        vcols = {c["name"]: c for c in insp.get_columns("reviewers")} if "reviewers" in tables else {}
        with engine.begin() as c:
            if "tmdb_id" not in mcols:
                c.execute(text("ALTER TABLE movies ADD COLUMN tmdb_id VARCHAR(32) NULL"))
            # TMDb 新增列
            if "poster" not in mcols:
                c.execute(text("ALTER TABLE movies ADD COLUMN poster LONGBLOB NULL"))
            if "source" not in mcols:
                c.execute(text("ALTER TABLE movies ADD COLUMN source VARCHAR(16) NULL"))
            if "tagline" not in mcols:
                c.execute(text("ALTER TABLE movies ADD COLUMN tagline VARCHAR(512) NULL"))
            # TMDb 热度值（用于热度排行/排序）
            if "popularity" not in mcols:
                c.execute(text("ALTER TABLE movies ADD COLUMN popularity FLOAT NULL"))
            # 删除豆瓣专属的评分分布列（TMDb 无此数据，留着会让接口返回一堆 NULL）
            for col in ("star_5", "star_4", "star_3", "star_2", "star_1"):
                if col in mcols:
                    c.execute(text(f"ALTER TABLE movies DROP COLUMN {col}"))
            # 删除豆瓣 id 列（纯 TMDb 化）
            if "douban_id" in mcols:
                c.execute(text("ALTER TABLE movies DROP COLUMN douban_id"))
            # TMDb 新增列：reviews.source（区分爬取的 tmdb 评论 / 用户发布的评论）
            if "source" not in rcols:
                c.execute(text("ALTER TABLE reviews ADD COLUMN source VARCHAR(16) NULL DEFAULT 'tmdb'"))
            if "tmdb_review_id" not in rcols:
                c.execute(text("ALTER TABLE reviews ADD COLUMN tmdb_review_id VARCHAR(64) NULL"))
            if "douban_review_id" in rcols:
                c.execute(text("ALTER TABLE reviews DROP COLUMN douban_review_id"))
            if "douban_url" in rcols:
                c.execute(text("ALTER TABLE reviews DROP COLUMN douban_url"))
            if "douban_uid" in vcols:
                c.execute(text("ALTER TABLE reviewers DROP COLUMN douban_uid"))
            # 删除短评表（短评是豆瓣独有概念，TMDb API 无此数据）
            if "short_comments" in tables:
                c.execute(text("DROP TABLE short_comments"))
    except Exception as e:  # noqa: BLE001
        # 无 MySQL 连接或权限不足时跳过
        print(f"[db] 结构迁移跳过：{e}")


_migrate_columns()
_ensure_database()
