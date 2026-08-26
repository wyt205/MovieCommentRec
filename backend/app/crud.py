from datetime import datetime

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app import models

# 国家/地区名 中文 → 数据库存储的英文（库里 country 字段存的是英文，如
# "United States of America" / "Japan" / "South Korea"）。用户用中文问「美国」
# 时先映射到英文再做 LIKE，否则 contains 永远匹配不到。
COUNTRY_ALIAS = {
    "美国": "United States of America",
    "usa": "United States of America",
    "美": "United States of America",
    "中国": "China",
    "大陆": "China",
    "中国大陆": "China",
    "中国香港": "Hong Kong",
    "香港": "Hong Kong",
    "中国台湾": "Taiwan",
    "台湾": "Taiwan",
    "日本": "Japan",
    "日": "Japan",
    "韩国": "South Korea",
    "南韩": "South Korea",
    "朝鲜": "North Korea",
    "英国": "United Kingdom",
    "英": "United Kingdom",
    "法国": "France",
    "法": "France",
    "德国": "Germany",
    "德": "Germany",
    "印度": "India",
    "俄": "Russia",
    "俄罗斯": "Russia",
    "意大利": "Italy",
    "西班牙": "Spain",
    "墨西哥": "Mexico",
    "加拿大": "Canada",
    "澳大利亚": "Australia",
    "新西兰": "New Zealand",
    "泰国": "Thailand",
}


# ---------------------- 电影 ----------------------
def get_movies(db: Session, skip: int = 0, limit: int = 20, keyword: str | None = None,
               genre: str | None = None, year: int | None = None,
               country: str | None = None, sort: str = "rating"):
    """电影列表，支持 关键词 / 类型 / 年份 / 国家地区 任意组合筛选与排序。

    sort: rating(按评分降序) | popularity(按热度降序) | release(按上映时间降序)
          | year(按年份降序) | recent(按入库时间降序)
    country 接受中文或英文，内部会先映射到库内英文存储再做 LIKE 匹配。
    """
    q = db.query(models.Movie)
    if keyword:
        q = q.filter(models.Movie.title.contains(keyword))
    if genre:
        # genres 存为 "动画, 爱情" 形式（逗号+空格）。用户可能一次提多个类型（逗号分隔，
        # 兼容中英文逗号 ",/，" 与前后空格），例如「动画,爱情」「科幻，喜剧」。
        # 多个类型之间取 AND（同时命中），契合「爱情的动画」这类交集诉求；
        # 每个类型单独做 LIKE，避免把整串当子串匹配时因空格/顺序而漏掉。
        parts = [g.strip() for g in str(genre).replace("，", ",").split(",") if g.strip()]
        for g in parts:
            q = q.filter(models.Movie.genres.contains(g))
    if year:
        q = q.filter(models.Movie.year == year)
    if country:
        country_en = COUNTRY_ALIAS.get(country.strip(), country.strip())
        q = q.filter(models.Movie.country.contains(country_en))
    if sort == "popularity":
        q = q.order_by(models.Movie.popularity.desc())
    elif sort == "release":
        q = q.order_by(models.Movie.release_date.desc())
    elif sort == "year":
        q = q.order_by(models.Movie.year.desc())
    elif sort == "recent":
        q = q.order_by(models.Movie.created_at.desc())
    else:  # rating 默认
        q = q.order_by(models.Movie.rating.desc())
    return q.offset(skip).limit(limit).all()


def count_movies(db: Session, genre: str | None = None, year: int | None = None,
                 country: str | None = None) -> int:
    """返回符合 类型/年份/地区 组合的「真实总条数」（不受 limit 影响）。

    与 get_movies 用同一套过滤逻辑，但只 COUNT 不返回行。供「诚实告知缺量」用——
    例如用户要 5 部、库里只有 4 部时，必须报真实的 4 而不是被 limit 截断后的数字。
    """
    q = db.query(models.Movie)
    if genre:
        parts = [g.strip() for g in str(genre).replace("，", ",").split(",") if g.strip()]
        for g in parts:
            q = q.filter(models.Movie.genres.contains(g))
    if year:
        q = q.filter(models.Movie.year == year)
    if country:
        country_en = COUNTRY_ALIAS.get(country.strip(), country.strip())
        q = q.filter(models.Movie.country.contains(country_en))
    return q.count()


def get_movie(db: Session, movie_id: int):
    return db.get(models.Movie, movie_id)


def get_movies_fuzzy(db: Session, keyword: str, limit: int = 10):
    """二元分词模糊匹配：处理『奇幻大冒险』→『奇幻变身大冒险』这类非连续子串。

    把关键词切成相邻 2 字（奇幻/幻大/大冒/冒险），标题命中任一即视为匹配，
    用 OR 合并。专治用户只记得片名片段、或记错个别字的情况。
    """
    if not keyword or len(keyword) < 2:
        return []
    grams = [keyword[i : i + 2] for i in range(len(keyword) - 1)]
    conds = [models.Movie.title.contains(g) for g in grams]
    return (
        db.query(models.Movie)
        .filter(or_(*conds))
        .order_by(models.Movie.rating.desc())
        .limit(limit)
        .all()
    )


def get_movie_by_tmdb_id(db: Session, tmdb_id: str):
    """按 TMDb id 查电影（供『仅爬取新数据』在拉详情前快速查重）。"""
    if not tmdb_id:
        return None
    return db.query(models.Movie).filter(models.Movie.tmdb_id == str(tmdb_id)).first()


def create_movie(db: Session, data: dict) -> models.Movie:
    """按 tmdb_id upsert（存在则更新，不存在则插入）。"""
    obj = None
    if data.get("tmdb_id"):
        obj = db.query(models.Movie).filter(models.Movie.tmdb_id == data["tmdb_id"]).first()
    if obj is None:
        obj = models.Movie()
        db.add(obj)
    for k, v in data.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    db.commit()
    db.refresh(obj)
    return obj


# ---------------------- 评论者 ----------------------
def get_or_create_reviewer(db: Session, name: str, avatar_url: str | None = None):
    obj = db.query(models.Reviewer).filter(models.Reviewer.name == name).first()
    if obj:
        return obj
    obj = models.Reviewer(name=name, avatar_url=avatar_url)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


# ---------------------- 影评 ----------------------
def get_reviews_by_movie(db: Session, movie_id: int, skip: int = 0, limit: int = 20):
    return (
        db.query(models.Review)
        .filter(models.Review.movie_id == movie_id)
        .order_by(models.Review.publish_date.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def create_review(db: Session, data: dict) -> models.Review:
    """按 tmdb_review_id upsert。"""
    obj = None
    if data.get("tmdb_review_id"):
        obj = db.query(models.Review).filter(
            models.Review.tmdb_review_id == data["tmdb_review_id"]).first()
    if obj is None:
        obj = models.Review()
        db.add(obj)
    for k, v in data.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    db.commit()
    db.refresh(obj)
    return obj


# ---------------------- 演员 ----------------------
def get_cast_by_movie(db: Session, movie_id: int) -> list:
    """返回某电影的演员列表（按 order 排序，主演在前）。"""
    return (
        db.query(models.MovieCast)
        .filter(models.MovieCast.movie_id == movie_id)
        .order_by(models.MovieCast.order)
        .all()
    )


def replace_movie_cast(db: Session, movie_id: int, cast_list: list[dict]) -> None:
    """整批替换某电影的演员（重爬时保证与 TMDb 一致，不累积重复）。"""
    db.query(models.MovieCast).filter(models.MovieCast.movie_id == movie_id).delete()
    for c in cast_list:
        db.add(models.MovieCast(movie_id=movie_id, **c))
    db.commit()


# ---------------------- 评分分布 ----------------------
def get_rating_distribution(db: Session, movie_id: int) -> dict:
    """统计某电影「有评分」的影评在各星级的分布（含 TMDb 与用户评论）。
    返回 { total, distribution:{1..5}, average }。无评分影评不计入。"""
    rows = (
        db.query(models.Review.rating, func.count())
        .filter(models.Review.movie_id == movie_id, models.Review.rating.isnot(None))
        .group_by(models.Review.rating)
        .all()
    )
    dist = {i: 0 for i in range(1, 6)}
    total = 0
    for r, cnt in rows:
        if isinstance(r, int) and 1 <= r <= 5:
            dist[r] += cnt
            total += cnt
    average = round(sum(k * v for k, v in dist.items()) / total, 2) if total else None
    return {"total": total, "distribution": dist, "average": average}


# ---------------------- 用户发布评论 ----------------------
_RATING_LABELS = {5: "力荐", 4: "推荐", 3: "还行", 2: "较差", 1: "很差"}


def create_user_review(db: Session, movie_id: int, nickname: str,
                       rating: int, content: str) -> models.Review:
    """用户在前端点星打分 + 写短评，入库为一条 source='user' 的评论。
    先按昵称复用评论者（轻量用户体系，无需登录）。"""
    reviewer = get_or_create_reviewer(db, name=nickname)
    rv = models.Review(
        movie_id=movie_id,
        reviewer_id=reviewer.id,
        title=f"{nickname}的短评",
        rating=rating,
        rating_label=_RATING_LABELS.get(rating),
        summary=content[:200],
        content=content,
        source="user",
        publish_date=datetime.now(),
    )
    db.add(rv)
    db.commit()
    db.refresh(rv)
    return rv
