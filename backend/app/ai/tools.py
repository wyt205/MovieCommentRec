"""Agent 工具：用 LangChain @tool 装饰器，把数据库查询封装成模型可调用的工具。

每个工具函数：
- 用中文 docstring 描述「它能做什么、参数是什么」——LangChain 会据此自动生成
  JSON Schema 交给模型，模型通过 function calling 决定何时调用、传什么参数。
- 内部自行打开/关闭数据库会话（SessionLocal），与 HTTP 请求链路解耦，便于独立测试。
- 返回值统一为「给人/模型读的纯文本」，模型再把这些信息组织成最终回答。
"""

from contextvars import ContextVar
from langchain_core.tools import tool

from app import crud
from app.db.database import SessionLocal

# 本轮用户原话（由 chat() 注入），用于「类型词自动抽取」——
# 弱 FC 模型在「X题材的Y」组合里极易只抽一个类型、丢掉其它，
# 故不再依赖模型分解，改为代码从用户原话里确定性地抽出所有类型词并合并过滤。
_USER_QUERY_CTX: ContextVar[str] = ContextVar("user_query", default="")

# TMDb 官方电影类型（中文），用于从用户原话里抽取类型。这些词互不为子串，安全。
GENRE_KEYWORDS = [
    "动作", "冒险", "动画", "喜剧", "犯罪", "纪录片", "剧情", "家庭",
    "奇幻", "历史", "恐怖", "音乐", "爱情", "科幻", "电视电影", "惊悚",
    "战争", "西部",
]
# 含否定词时保守地不做自动类型抽取，避免把「不要动画」误判为要动画
_NEGATION_WORDS = ["不要", "非", "别", "不是", "除了", "排除", "不想要", "不想看", "别推荐"]


# 口语 / 同义词 → 标准 TMDb 类型。用户和弱模型常说「刺激 / 搞笑 / 感人」等，
# 但库里存的是标准类型（动作 / 喜剧 / 剧情…）。归一到标准类型才能让查询命中。
# 注意：每个口语词只映射到「单一」标准类型——因为 crud 的多类型是 AND 关系，
# 映射到多个会过于严苛（几乎查不到）。取最贴合的那一个即可。
_GENRE_SYNONYMS = {
    "刺激": "动作",
    "热血": "动作",
    "打斗": "动作",
    "燃": "动作",
    "爽": "动作",
    "搞笑": "喜剧",
    "幽默": "喜剧",
    "搞怪": "喜剧",
    "感人": "剧情",
    "催泪": "剧情",
    "温暖": "剧情",
    "治愈": "剧情",
    "励志": "剧情",
    "浪漫": "爱情",
    "甜": "爱情",
    "吓人": "恐怖",
    "惊悚向": "惊悚",
    "悬疑": "惊悚",
    "烧脑": "科幻",
}


def _normalize_genre_token(token: str) -> str | None:
    """把单个类型词归一到标准 TMDb 类型；既不是标准类型也不是已知同义词则返回 None（丢弃）。"""
    if not token:
        return None
    if token in GENRE_KEYWORDS:
        return token
    if token in _GENRE_SYNONYMS:
        return _GENRE_SYNONYMS[token]
    return None


def _detect_genres_from_text(text: str) -> list[str]:
    """从一段文本里确定性地抽出所有出现的电影类型词（含口语同义词），并归一到标准类型。"""
    if not text:
        return []
    if any(neg in text for neg in _NEGATION_WORDS):
        return []
    found: list[str] = []
    # 先匹配口语同义词（归一为标准类型），再匹配标准类型本身；均去重
    for syn, real in _GENRE_SYNONYMS.items():
        if syn in text and real not in found:
            found.append(real)
    for g in GENRE_KEYWORDS:
        if g in text and g not in found:
            found.append(g)
    return found


def _merge_genres(genre_arg: str) -> str | None:
    """把模型传入的 genre 与「用户原话自动抽取的类型」合并去重，逗号分隔。

    模型可能传标准类型（动作）、口语同义词（刺激→动作）、或完全不是类型的词。
    这里统一归一：标准类型保留、口语词映射到标准类型、既不是标准也不是同义词的丢弃
    （避免「刺激」这种非标准词进入 AND 过滤把整个查询杀成 0 结果）。
    返回 None 表示无任何类型条件（不过滤类型）。
    """
    explicit_raw = [g.strip() for g in str(genre_arg or "").replace("，", ",").split(",") if g.strip()]
    norm_explicit: list[str] = []
    for g in explicit_raw:
        norm = _normalize_genre_token(g)
        if norm and norm not in norm_explicit:
            norm_explicit.append(norm)
    auto = _detect_genres_from_text(_USER_QUERY_CTX.get())
    merged = norm_explicit + [g for g in auto if g not in norm_explicit]
    return ",".join(merged) if merged else None


@tool
def search_movies(keyword: str) -> str:
    """按电影名称（可不完整）搜索电影。参数 keyword 为用户想找的片名，例如只记得「奇幻大冒险」也能通过模糊匹配到《奇幻变身大冒险》。注意：本工具只做片名匹配，若想按「剧情/主题/感受」找电影（如"讲时间循环的"），请用 semantic_search_movies。"""
    db = SessionLocal()
    try:
        # 1) 先精确子串匹配
        rows = crud.get_movies(db, keyword=keyword, limit=10)
        # 2) 精确无果 → 二元分词模糊匹配（治「记错/只记得片段」）
        if not rows:
            rows = crud.get_movies_fuzzy(db, keyword, limit=10)
        # 3) 仍无果 → 直接判定资料库暂无该片（不做语义兜底，避免为「查无此片」多等一次网络往返，也更直接）
        if not rows:
            return f"未找到与「{keyword}」相关的电影。"
        lines = [f"找到 {len(rows)} 部相关电影："]
        for m in rows:
            lines.append(
                f"- 《{m.title}》（{m.year or '未知'}）评分:{m.rating or '无'} "
                f"类型:{m.genres or '未知'} id={m.id}"
            )
        return "\n".join(lines)
    finally:
        db.close()


@tool
def find_movies(genre: str = "", year: int = 0, country: str = "",
                sort: str = "rating", limit: int = 5) -> str:
    """按任意条件组合筛选并排序电影，是「找电影/推荐电影」的统一入口。所有参数均可选、可任意组合，模型应根据用户的话挑出相关参数填入：
    - genre: 类型名（中文，如 科幻、喜剧、动作、动画、剧情、恐怖、爱情）。**注意：类型这一项已由系统在拿到用户原话后自动抽取并合并过滤（代码级、确定性），你即使只传其中一个类型，系统也会把用户话里提到的其它类型一并加上、取交集。所以你只管正常传参即可，不必为「多个类型」反复纠结。**
    - year: 上映年份（数字，如 2021）
    - country: 国家/地区（中文，如 美国、日本、韩国、中国），内部会自动映射到库内英文存储
    - sort: 排序方式——rating(按评分降序,默认) / popularity(按热度降序) / year(按年份降序) / release(按上映日期降序)
    - limit: 返回条数（默认5，最多20）
    适用示例：「评分最高的动作片」→ genre=动作, sort=rating；「2021年上映的中国电影」→ year=2021, country=中国；「按热度排的科幻片」→ genre=科幻, sort=popularity；「评分最高且是2024年的」→ year=2024, sort=rating；「爱情的动画」→ 系统会自动锁定 动画 AND 爱情。**绝不要自行脑补用户没说的类型维度（年份/地区）；用户没提就留空。**"""
    db = SessionLocal()
    try:
        # 类型条件 = 模型传入的 genre + 用户原话自动抽取的类型（代码级合并，治弱模型漏掉多类型）
        genre_merged = _merge_genres(genre)
        rows = crud.get_movies(
            db,
            genre=genre_merged,
            year=year or None,
            country=country or None,
            sort=sort or "rating",
            limit=min(limit or 5, 20),
        )
        if not rows:
            cond = "、".join(
                f"{k}={v}" for k, v in
                [("类型", genre_merged or ""), ("年份", year or ""), ("地区", country)]
                if v
            )
            return f"未找到符合条件（{cond or '全部'}）的电影。"
        # 真实总条数（不受 limit 截断影响），供「诚实告知缺量」使用——
        # 例如库里实际有 20 部动画、用户只要 2 部时，这里报 20 而非被截断后的 2。
        total = crud.count_movies(
            db, genre=genre_merged, year=year or None, country=country or None
        )
        lines = [f"找到 {len(rows)} 部电影（按{sort}排序，共 {total} 部符合条件）："]
        for m in rows:
            pop = f" 热度:{m.popularity}" if m.popularity is not None else ""
            lines.append(
                f"- 《{m.title}》（{m.year or '未知'}）评分:{m.rating or '无'}{pop} "
                f"类型:{m.genres or '未知'} 地区:{m.country or '未知'} id={m.id}"
            )
        return "\n".join(lines)
    finally:
        db.close()


@tool
def get_movie_detail(movie_id: int) -> str:
    """获取某部电影的详细信息，包括导演、演员、剧情简介、评分、上映日期、类型等。参数 movie_id 为电影的数字 id（来自 search_movies / find_movies / get_similar_movies 返回结果里的 id 字段，不要自己编造）。"""
    db = SessionLocal()
    try:
        m = crud.get_movie(db, movie_id)
        if not m:
            return f"未找到 id={movie_id} 的电影。"
        cast = crud.get_cast_by_movie(db, movie_id)
        cast_str = "、".join(c.name for c in cast[:8]) or "未知"
        return (
            f"《{m.title}》（{m.original_title or ''} {m.year or ''}）\n"
            f"导演：{m.directors or '未知'}\n"
            f"演员：{cast_str}\n"
            f"类型：{m.genres or '未知'}\n"
            f"评分：{m.rating or '无'}\n"
            f"上映日期：{m.release_date or '未知'}\n"
            f"国家/地区：{m.country or '未知'}\n"
            f"简介：{m.summary or '暂无简介'}\n"
            f"宣传语：{m.tagline or '无'}"
        )
    finally:
        db.close()


@tool
def get_movie_info_by_name(name: str) -> str:
    """根据电影名（可不完整，如只记得「奇幻大冒险」也能匹配《奇幻变身大冒险》）直接获取该电影的详细资料：导演、演员、类型、评分、上映日期、简介、宣传语等。当用户提到某部具体电影、想看它的详情时使用——直接传片名即可，无需先查 id，也绝不要自己编造 id。"""
    db = SessionLocal()
    try:
        rows = crud.get_movies(db, keyword=name, limit=5)
        if not rows:
            rows = crud.get_movies_fuzzy(db, name, limit=5)
        if not rows:
            return f"未找到与「{name}」相关的电影。"
        m = rows[0]
        cast = crud.get_cast_by_movie(db, m.id)
        cast_str = "、".join(c.name for c in cast[:8]) or "未知"
        return (
            f"《{m.title}》（{m.original_title or ''} {m.year or ''}）\n"
            f"导演：{m.directors or '未知'}\n"
            f"演员：{cast_str}\n"
            f"类型：{m.genres or '未知'}\n"
            f"评分：{m.rating or '无'}\n"
            f"上映日期：{m.release_date or '未知'}\n"
            f"国家/地区：{m.country or '未知'}\n"
            f"简介：{m.summary or '暂无简介'}\n"
            f"宣传语：{m.tagline or '无'}"
        )
    finally:
        db.close()


@tool
def get_similar_movies(movie_id: int, limit: int = 5) -> str:
    """获取与某部电影相似的电影（基于相同类型、按评分排序）。参数 movie_id 为电影数字 id。"""  # noqa: E501
    db = SessionLocal()
    try:
        m = crud.get_movie(db, movie_id)
        if not m:
            return f"未找到 id={movie_id} 的电影。"
        genre = ""
        if m.genres:
            genre = m.genres.replace("，", ",").split(",")[0].strip()
        if not genre:
            return f"《{m.title}》暂无类型信息，无法推荐相似电影。"
        rows = crud.get_movies(db, genre=genre, sort="rating", limit=limit + 1)
        similar = [x for x in rows if x.id != m.id][:limit]
        if not similar:
            return f"未找到与《{m.title}》相似的电影。"
        lines = [f"与《{m.title}》相似的电影："]
        for x in similar:
            lines.append(f"- 《{x.title}》（{x.year or '未知'}）评分:{x.rating or '无'} id={x.id}")
        return "\n".join(lines)
    finally:
        db.close()


@tool
def get_reviews(movie_id: int, limit: int = 5) -> str:
    """获取某部电影的影评列表（标题与摘要）。参数 movie_id 为电影数字 id。"""
    db = SessionLocal()
    try:
        rows = crud.get_reviews_by_movie(db, movie_id, limit=limit)
        if not rows:
            return "该电影暂没有影评。"
        lines = [f"该电影共有 {len(rows)} 条影评（展示前 {len(rows)} 条）："]
        for r in rows:
            lines.append(f"- 《{r.title}》（{r.rating_label or ''} {r.rating or ''}星）{r.summary or ''}")
        return "\n".join(lines)
    finally:
        db.close()


@tool
def semantic_search_movies(query: str, top_k: int = 5) -> str:
    """按语义（剧情/主题/情感）搜索电影，而不是按片名关键词。参数 query 为用户对电影的自然语言描述，例如「一部讲时间循环的科幻片」「结局治愈、让人流泪的电影」「适合深夜一个人看的小众文艺片」。当用户用含义/感受/主题来描述想看的电影（关键词搜不到）时，优先使用本工具。"""
    try:
        from app.ai.rag import semantic_search, SEARCH_THRESHOLD
    except Exception as e:  # noqa: BLE001
        return f"语义检索暂不可用：{e}"
    try:
        results = semantic_search(query, top_k=top_k)
    except Exception as e:  # noqa: BLE001
        return f"语义检索执行出错：{type(e).__name__}: {e}"
    if not results:
        return "语义检索库为空，请先运行建库脚本（build_embeddings）或电影表暂无数据。"
    # 置信度阈值：top1 都低于阈值说明库里没有足够相关的电影，诚实告知而非硬塞不沾边的
    top_score, _ = results[0]
    if top_score < SEARCH_THRESHOLD:
        return (
            f"未找到与「{query}」高度相关的电影（最相近的也仅相关度 {top_score:.2f}）。\n"
            f"建议：换一个更具体的描述，或用「按电影名称或简介搜索电影」工具直接用关键词查找。"
        )
    lines = [f"语义检索「{query}」找到 {len(results)} 部相关电影："]
    for score, m in results:
        # 仅展示达到置信度阈值的结果，避免末尾出现明显不相关的电影
        if score < SEARCH_THRESHOLD:
            break
        lines.append(
            f"- 《{m.title}》（{m.year or '未知'}）评分:{m.rating or '无'} 相关度:{score:.2f} id={m.id}"
        )
    return "\n".join(lines)


# 工具清单（装配 Agent 时使用）
TOOLS = [search_movies, find_movies, get_movie_detail, get_movie_info_by_name, get_similar_movies, get_reviews, semantic_search_movies]
