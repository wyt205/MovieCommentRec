"""
TMDb 官方电影类型（genres）单一数据源
=================================================================
TMDb 官方固定返回这 19 个电影类型（带稳定 id）。本项目所有「分类」
都以此为准：
  - 爬虫按 tmdb genre_id 映射成这里的中文名存库；
  - 后端 /api/genres 直接返回本列表作为「分类目录」；
  - 前端分类标签从 /api/genres 拉取，不写死，保证严格一致。

官方列表参考：https://developer.themoviedb.org/docs/genres （movie genres）
"""
from __future__ import annotations

# TMDb 官方电影类型（id 为 TMDb 固定值，name 为项目内统一中文名）
TMDB_GENRES: list[dict] = [
    {"id": 28, "name": "动作"},
    {"id": 12, "name": "冒险"},
    {"id": 16, "name": "动画"},
    {"id": 35, "name": "喜剧"},
    {"id": 80, "name": "犯罪"},
    {"id": 99, "name": "纪录片"},
    {"id": 18, "name": "剧情"},
    {"id": 10751, "name": "家庭"},
    {"id": 14, "name": "奇幻"},
    {"id": 36, "name": "历史"},
    {"id": 27, "name": "恐怖"},
    {"id": 10402, "name": "音乐"},
    {"id": 9648, "name": "悬疑"},
    {"id": 10749, "name": "爱情"},
    {"id": 878, "name": "科幻"},
    {"id": 10770, "name": "电视电影"},
    {"id": 53, "name": "惊悚"},
    {"id": 10752, "name": "战争"},
    {"id": 37, "name": "西部"},
]

# id -> 中文名 的快速映射（爬虫用，保证存库名与目录完全一致）
GENRE_ID_TO_NAME: dict[int, str] = {g["id"]: g["name"] for g in TMDB_GENRES}

# 中文名集合（用于校验/去重）
GENRE_NAMES: set[str] = {g["name"] for g in TMDB_GENRES}


def genre_name(genre_id: int | None) -> str | None:
    """把 TMDb genre_id 转成项目统一中文名；未知 id 返回 None。"""
    if genre_id is None:
        return None
    return GENRE_ID_TO_NAME.get(int(genre_id))
