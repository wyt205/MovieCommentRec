from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.sql import func

from app.db.database import Base


class MovieCast(Base):
    """电影演员表（结构化：演员名 + 饰演角色 + 头像路径）。
    由爬虫从 TMDb /movie/{id}/credits 的 cast 写入，详情页据此渲染「演员表」。
    profile_path 走 /api/cast-photo 代理按需加载（国内免访问 image.tmdb.org）。"""

    __tablename__ = "movie_cast"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    movie_id = Column(BigInteger, ForeignKey("movies.id", ondelete="CASCADE"), nullable=False, index=True)
    tmdb_person_id = Column(String(32))
    name = Column(String(128), nullable=False, comment="演员名")
    character = Column(String(128), comment="饰演角色")
    order = Column(Integer, default=0, comment="排序（主演在前）")
    profile_path = Column(String(255), comment="TMDb 头像路径（按需代理）")


class Movie(Base):
    __tablename__ = "movies"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tmdb_id = Column(String(32), unique=True, nullable=True, comment="TMDb movie id（唯一来源）")
    title = Column(String(255), nullable=False)
    original_title = Column(String(255))
    year = Column(Integer)
    directors = Column(String(512))
    writers = Column(String(512))
    casts = Column(String(1024))
    genres = Column(String(255))
    country = Column(String(255))
    language = Column(String(255))
    release_date = Column(String(255))
    runtime = Column(String(128))
    aka = Column(String(512))
    imdb = Column(String(32))
    rating = Column(Float)
    rating_count = Column(Integer)
    # TMDb 热度值（popularity），用于「热度排行 / 热度排序」
    popularity = Column(Float)
    summary = Column(Text)
    poster_url = Column(String(512))
    # 海报图片字节（不再落盘到文件夹，直接存库，国内也能正常显示）
    poster = Column(LargeBinary)
    # 数据来源（当前固定 tmdb）
    source = Column(String(16))
    # TMDb 宣传语（tagline），如「希望让人自由」
    tagline = Column(String(512))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class MovieEmbedding(Base):
    """电影语义向量（RAG 用）。每部电影一条，embedding 为 768 维向量（JSON 存储）。
    由 build_embeddings.py 批量生成；语义检索时全部载入内存做余弦相似度。"""

    __tablename__ = "movie_embeddings"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    movie_id = Column(BigInteger, ForeignKey("movies.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    chunk_text = Column(Text, comment="被嵌入的文本（标题+类型+导演+简介等）")
    embedding = Column(JSON, comment="768 维向量")
    model = Column(String(64), comment="嵌入模型 id")
    created_at = Column(DateTime, server_default=func.now())


class Reviewer(Base):
    __tablename__ = "reviewers"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False)
    avatar_url = Column(String(512))
    location = Column(String(128))
    signature = Column(String(512))
    created_at = Column(DateTime, server_default=func.now())


class Genre(Base):
    """电影类型目录（分类的单一数据源），内容由 app/core/genres.py 的
    TMDb 官方 19 个类型播种而来。便于分类/筛选与前端拉取分类标签。"""

    __tablename__ = "genres"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tmdb_id = Column(Integer, unique=True, nullable=False, comment="TMDb 类型 id")
    name = Column(String(64), nullable=False, comment="类型中文名")


class Review(Base):
    __tablename__ = "reviews"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tmdb_review_id = Column(String(64), unique=True, nullable=True, comment="TMDb review id（唯一来源）")
    movie_id = Column(BigInteger, ForeignKey("movies.id", ondelete="CASCADE"), nullable=False)
    reviewer_id = Column(BigInteger, ForeignKey("reviewers.id", ondelete="SET NULL"))
    source = Column(String(16), default="tmdb", comment="评论来源：tmdb（爬取）/ user（用户发布）")
    title = Column(String(512), nullable=False)
    rating = Column(Integer)
    rating_label = Column(String(8))
    summary = Column(Text)
    content = Column(Text)
    useful_count = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)
    views = Column(Integer, default=0)
    publish_date = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class AgentTrace(Base):
    """Agent 对话埋点：每次 /api/agent/chat 与 /api/admin/chat 都落一条，
    供管理端做「日志 / 缓存命中率 / 护栏使用率 / 评测」可视化。

    - tool_calls：本次对话模型自主调用的工具清单（名称+参数），JSON 存储。
    - used_guardrail：是否触发了「涉及电影却没调工具→强制重试」的护栏。
    - cache_hit：是否命中进程内响应缓存（同问题短期复问直接复用答案）。
    - latency_ms：端到端耗时（含 LLM + 工具）。
    """

    __tablename__ = "agent_traces"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True, comment="会话 id")
    query = Column(Text, comment="用户原话")
    intent = Column(String(32), comment="代码判定的意图：chat/identity/defer/movie_name/recommend/movie_related/cache_hit")
    tool_calls = Column(JSON, comment="模型自主调用的工具清单 [{name,args}]")
    answer = Column(Text, comment="助手最终回复")
    latency_ms = Column(Integer, comment="端到端耗时(毫秒)")
    used_guardrail = Column(Boolean, default=False, comment="是否触发护栏强制重试")
    cache_hit = Column(Boolean, default=False, comment="是否命中响应缓存")
    created_at = Column(DateTime, server_default=func.now())


class ChatSession(Base):
    """Agent 对话会话：每次「新建对话」一条记录，messages 存完整对话历史（JSON）。
    由 agent 的 DBChatMessageHistory 自动读写，因此切换页面 / 重启前后端都不丢，
    并支撑前端「左侧对话记录列表」（可继续聊 / 删除）。"""

    __tablename__ = "chat_sessions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True, comment="会话唯一 id（前端持有）")
    title = Column(String(255), default="新对话", comment="会话标题（取首条用户消息前 24 字）")
    messages = Column(JSON, default=list, comment="对话历史 [{role, text, ts}]")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class UserPreference(Base):
    """用户长期偏好（轻量 KV）：目前存电影类型喜好，供 agent 推荐时主动参考。
    key 如 fav_genres / disliked_genres，value 为字符串列表。"""

    __tablename__ = "user_preferences"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    key = Column(String(64), unique=True, nullable=False)
    value = Column(JSON, nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
