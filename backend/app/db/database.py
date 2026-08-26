from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session, sessionmaker, declarative_base

from app.core.config import settings

# 连接 MySQL（PyMySQL 驱动）；pool_pre_ping 自动检测断连
engine = create_engine(settings.database_url, pool_pre_ping=True)

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
