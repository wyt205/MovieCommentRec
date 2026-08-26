from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator


# ---------------------- 电影 ----------------------
class MovieOut(BaseModel):
    id: int
    title: str
    year: Optional[int] = None
    directors: Optional[str] = None
    genres: Optional[str] = None
    rating: Optional[float] = None
    rating_count: Optional[int] = None
    popularity: Optional[float] = None
    poster_url: Optional[str] = None
    summary: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class MovieDetailOut(MovieOut):
    original_title: Optional[str] = None
    writers: Optional[str] = None
    casts: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = None
    release_date: Optional[str] = None
    runtime: Optional[str] = None
    aka: Optional[str] = None
    imdb: Optional[str] = None
    rating_count: Optional[int] = None
    source: Optional[str] = None
    tagline: Optional[str] = None
    reviews: list["ReviewOut"] = []
    cast: list["CastOut"] = []
    rating_distribution: Optional[dict] = None
    review_count: int = 0


# ---------------------- 演员 ----------------------
class CastOut(BaseModel):
    id: int
    name: str
    character: Optional[str] = None
    order: int = 0
    profile_path: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ---------------------- 评论者 ----------------------
class ReviewerOut(BaseModel):
    id: int
    name: str
    avatar_url: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ---------------------- 影评 ----------------------
class ReviewOut(BaseModel):
    id: int
    movie_id: int
    reviewer_id: Optional[int] = None
    title: str
    rating: Optional[int] = None
    rating_label: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None
    useful_count: int = 0
    comments_count: int = 0
    views: int = 0
    publish_date: Optional[datetime] = None
    movie_title: Optional[str] = None
    reviewer_name: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


ANONYMOUS_NAME = "匿名用户"


# ---------------------- 用户发布评论（带星级） ----------------------
class ReviewCreate(BaseModel):
    movie_id: int
    nickname: Optional[str] = ANONYMOUS_NAME
    rating: int  # 1-5 星
    content: str

    @field_validator("rating")
    @classmethod
    def _check_rating(cls, v: int) -> int:
        if not (1 <= v <= 5):
            raise ValueError("评分需在 1-5 之间")
        return v

    @field_validator("nickname")
    @classmethod
    def _fallback_nickname(cls, v: Optional[str]) -> str:
        s = (v or "").strip()
        return s or ANONYMOUS_NAME

    @field_validator("content")
    @classmethod
    def _strip_nonempty(cls, v: str) -> str:
        s = (v or "").strip()
        if not s:
            raise ValueError("评论内容不能为空")
        return s
