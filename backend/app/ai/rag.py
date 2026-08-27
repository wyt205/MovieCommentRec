"""RAG 核心：建库（build）+ 语义检索（search）。

设计（与用户确认）：向量直接存进现有 MySQL 的 movie_embeddings 表（JSON 列存 768 维），
检索时在 Python 里算余弦相似度。电影量几千部内毫秒级，零新依赖、与现有库一体。

为什么不用向量数据库：本项目是 demo 规模，MySQL + Python 余弦足够，且避免引入 Chroma
等额外服务与依赖，部署更省事。
"""

import datetime
import time

from sqlalchemy import select, text

import numpy as np

from app import models
from app.ai.embeddings import (
    embed_query,
    embed_texts,
    get_embedder,
    _embed_with_retry,
)
from app.core.config import settings
from app.db.database import Base, SessionLocal, engine

# 星火 Qwen3-Embedding-8B 实测维度
EMBED_DIM = 768
# 批量嵌入的批大小（避免一次性发太多文本）
BATCH = 32


def _ensure_table():
    """确保 movie_embeddings 表存在（建库脚本/独立运行时不经过 FastAPI 启动，
    不会自动 create_all，这里兜底创建，已存在则跳过）。

    同时幂等补齐 updated_at 列：create_all 只会建缺失的表、不会 ALTER 已有表，
    老库上的 movie_embeddings 没有该列，这里用原生 ALTER 补上（列已存在则忽略）。
    updated_at 用 MySQL 的 ON UPDATE CURRENT_TIMESTAMP，使每次 upsert 重建都自动刷新，
    用户可凭 SELECT MAX(updated_at) 肉眼确认「建库确实跑过、何时跑的」。
    """
    Base.metadata.create_all(bind=engine, tables=[models.MovieEmbedding.__table__], checkfirst=True)
    try:
        with engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE movie_embeddings "
                "ADD COLUMN updated_at DATETIME NOT NULL "
                "DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
            ))
    except Exception:  # noqa: BLE001
        # 列已存在（1060）或方言差异 —— 忽略，不阻断建库
        pass
    else:
        # ADD 首次成功（老库补列）：已带 ON UPDATE，无需再处理
        return
    # ADD 失败（列已存在）：旧 schema 可能建出「无 ON UPDATE」的列，补一句 MODIFY 兜底
    try:
        with engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE movie_embeddings "
                "MODIFY COLUMN updated_at DATETIME NOT NULL "
                "DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
            ))
    except Exception:  # noqa: BLE001
        pass


def _chunk_for_movie(m: models.Movie) -> str:
    """把一部电影拼成一段便于语义检索的文本。"""
    parts: list[str] = []
    if m.title:
        parts.append(m.title)
    if m.original_title:
        parts.append(m.original_title)
    if m.genres:
        parts.append("类型：" + m.genres)
    if m.directors:
        parts.append("导演：" + m.directors)
    if m.year:
        parts.append(f"年份：{m.year}")
    if m.tagline:
        parts.append(m.tagline)
    if m.summary:
        parts.append(m.summary)
    return "\n".join(p for p in parts if p)


def build_movie_embeddings(model_id: str | None = None) -> int:
    """遍历 movies 表，为每部电影生成语义向量并 upsert 进 movie_embeddings。

    可重复运行：已存在的 movie_id 会覆盖更新（换模型后重跑即可重建索引）。
    返回处理的电影数量。

    健壮性（关键，踩过坑）：
      星火 MaaS 免费档在限流/抖动时可能【静默返回少于请求的向量】
      （例如请求 32 条只回 28 条且不报错）。旧实现直接 zip(ids, vecs) 会被
      截断，导致「只有前 N 部电影被更新」，且因失败静默、launcher 仍报成功，
      极难排查（表现为 updated_at 时间簇分散、部分电影语义检索搜不到）。
      本实现逐小批嵌入并【严格校验返回长度】：长度不符则重试整批，仍不符则
      拆成单条逐个补齐；最终若仍有缺失则整体抛错（让 launcher 如实报「构建失败」，
      而非悄悄只更新一部分）。commit 前对所有行统一打 build_time 戳，使一次
      完整建库后 40 行 updated_at 完全一致，肉眼即可确认「本次覆盖全部」。
    """
    model_id = model_id or settings.embedding_model or "xop3qwen8bembedding"
    _ensure_table()
    db = SessionLocal()
    try:
        movies = db.query(models.Movie).all()
        existing = {row.movie_id: row for row in db.query(models.MovieEmbedding).all()}

        texts = [_chunk_for_movie(m) for m in movies]
        ids = [m.id for m in movies]

        # —— 健壮嵌入：保证 all_vecs 与 texts 等长，否则整体失败，绝不静默部分成功 ——
        embedder = get_embedder()
        all_vecs: list = [None] * len(texts)
        pending = list(range(len(texts)))
        for _round in range(4):                      # 整轮重试（应对整体限流）
            if not pending:
                break
            next_pending: list[int] = []
            for s in range(0, len(pending), BATCH):
                seg = pending[s : s + BATCH]
                seg_texts = [texts[i] for i in seg]
                try:
                    vecs = _embed_with_retry(embedder, seg_texts)
                except Exception:                    # 整批失败 → 留到下一轮
                    next_pending.extend(seg)
                    continue
                if vecs is None or len(vecs) != len(seg_texts):
                    # 长度不符（静默截断/部分返回）→ 逐条补齐，确保不丢任何一部
                    for i in seg:
                        try:
                            all_vecs[i] = _embed_with_retry(embedder, [texts[i]])[0]
                        except Exception:
                            next_pending.append(i)
                    continue
                for i, v in zip(seg, vecs):
                    all_vecs[i] = v
            pending = next_pending
            if pending:
                time.sleep(2)
        missing = [ids[i] for i, v in enumerate(all_vecs) if v is None]
        if missing:
            raise RuntimeError(
                f"建库失败：{len(missing)} 部电影嵌入未成功（疑似星火接口限流/抖动）："
                f"movie_id={missing[:10]}{'…' if len(missing) > 10 else ''}"
            )

        build_time = datetime.datetime.now()
        for mid, vec in zip(ids, all_vecs):
            row = existing.get(mid)
            if row is None:
                row = models.MovieEmbedding(movie_id=mid)
                db.add(row)
            row.embedding = vec
            row.chunk_text = _chunk_for_movie(db.get(models.Movie, mid))
            row.model = model_id
            row.updated_at = build_time          # 统一戳：本次建库覆盖全部 40 部

        db.commit()
        return len(ids)
    finally:
        db.close()


def _lexical_score(query: str, chunk: str) -> float:
    """轻量关键词命中分（归一化到 [0,1]），作为向量检索的兜底。

    设计动机：星火 MaaS 嵌入端点对「短关键词查询 vs 长文档」的检索对齐较弱，
    纯向量检索会把「鬼灭之刃」误判得不如某些恐怖片近。关键词命中能可靠地
    把标题/简介里真含该词条的 movies 顶上来——这正是企业级 RAG 常用的
    「稠密向量 + 稀疏关键词」混合检索思路。
    """
    q = (query or "").strip().lower()
    if not q or not chunk:
        return 0.0
    c = chunk.lower()
    if q in c:                      # 完整子串命中（如标题含「鬼灭之刃」）→ 最强
        return 1.0
    # 否则按字符覆盖：query 去重字符中有多少出现在 chunk（忽略无语义虚词）
    stop = set("的了吗呢啊吧哟嘛是和与及等")
    qs = {ch for ch in q if ch not in stop}
    if not qs:
        return 0.0
    hits = sum(1 for ch in qs if ch in c)
    return min(1.0, hits / len(qs))


# 混合检索权重：向量余弦为主，关键词命中兜底
DENSE_WEIGHT = 0.6
LEX_WEIGHT = 0.4

# 置信度阈值：top1 融合分低于此值视为「未找到高度相关电影」，避免硬塞不沾边的。
# 选定依据（explore_threshold.py 实测，库 40 部电影，混合权重 0.6/0.4）：
#   合理命中组 0.548~0.731，无关/误排组 0.361~0.538，0.54 正好将两组干净分开
#   （可挡掉「做蛋糕教程」「时间旅行科幻」等库里根本不存在的误排，同时保留擦边合理的「赛博朋克」）。
# 注意：该阈值依赖当前语料与嵌入模型；扩充电影量或换嵌入模型后需重跑 explore_threshold 校准。
SEARCH_THRESHOLD = 0.54


def semantic_search(query: str, top_k: int = 5) -> list[tuple[float, models.Movie]]:
    """对 query 做混合语义检索，返回 [(相关度, Movie), ...] 按相关度降序。

    实现（混合检索）：
      1. 向量侧：query 经检索指令前缀嵌入 → 与库中全部电影向量算余弦；
      2. 关键词侧：query 是否字面命中电影标题/简介（兜底防误排）；
      3. 融合分 = DENSE_WEIGHT*余弦 + LEX_WEIGHT*关键词分，取 top_k。
    """
    db = SessionLocal()
    try:
        _ensure_table()
        rows = db.query(models.MovieEmbedding).all()
        if not rows:
            return []

        qvec = np.array(embed_query(query), dtype=np.float32)
        scored: list[tuple[float, int]] = []
        for r in rows:
            vec = r.embedding
            if not vec or len(vec) != EMBED_DIM:
                continue
            v = np.array(vec, dtype=np.float32)
            cos = float(
                np.dot(qvec, v) / (np.linalg.norm(qvec) * np.linalg.norm(v) + 1e-9)
            )
            lex = _lexical_score(query, r.chunk_text or "")
            blended = DENSE_WEIGHT * cos + LEX_WEIGHT * lex
            scored.append((blended, r.movie_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[tuple[float, models.Movie]] = []
        for score, mid in scored[:top_k]:
            m = db.get(models.Movie, mid)
            if m:
                out.append((score, m))
        return out
    finally:
        db.close()
