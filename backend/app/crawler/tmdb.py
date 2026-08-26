"""
TMDb 数据源（公开 REST API，无反爬）
=================================================================
替代被豆瓣封锁的爬虫：TMDb (The Movie Database) 提供官方免费 API，
返回干净 JSON，无反爬，支持 40+ 语言（含中文）。本模块把 TMDb
电影 / 影评 ETL 进现有 movies / reviews 表（用 tmdb_id 列区分来源），
并把海报图片字节直接存进 movies.poster（BLOB 字段，不放文件夹），
由 /api/movies/{id}/poster 接口回传，国内也能正常显示。

鉴权（两种都支持，自动识别，二选一即可）：
  - v4 Read Access Token（JWT，形如 eyJ...）：用 `Authorization: Bearer <token>` 头
  - v3 API Key（32 位 hex）：作为 `api_key` 查询参数
  启动器面板 / 命令行统一叫 "API Key 或 Token"，用户粘贴哪种都行。

网络（中国大陆必看）：
  - api.themoviedb.org 在国内通常直连不通，需要代理/VPN。
  - 本模块会自动读取环境变量 HTTPS_PROXY/HTTP_PROXY/ALL_PROXY；
  - 在 Windows 上还会自动读取「系统设置 → 代理」里的地址；
  - 也可在启动器「代理」框手动填入，例如 http://127.0.0.1:7890。

使用：
  - 启动器「数据爬取」面板填 Key（留空则用 .env）/ 数量 / 可选关键词 / 可选代理 → 点开始；
  - 或命令行：
        TMDB_API_KEY=xxxx python -m app.crawler.tmdb
        TMDB_COUNT=20 TMDB_SEARCH=盗梦空间 HTTPS_PROXY=http://127.0.0.1:7890 python -m app.crawler.tmdb
  - Key 免费申请：https://www.themoviedb.org → 设置 → API → Developer
注意事项：
  - 海报不落盘：字节直接存入 movies.poster（BLOB），由 /api/movies/{id}/poster 回传，
    国内无需访问 image.tmdb.org 即可显示封面（下载失败时 poster_url 退回 TMDb 远程链接）。
  - TMDb 影评以英文为主（用户生态），中文影评较少；电影元信息可中文。
"""
from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime

import requests
from app.db.database import SessionLocal
from app import crud
from app.core.genres import genre_name

# 基础地址可经环境变量 TMDB_BASE_URL 覆盖（例如指向可用的镜像/代理前缀）
TMDB_V3 = os.environ.get("TMDB_BASE_URL", "https://api.themoviedb.org/3").rstrip("/")
IMG_BASE = "https://image.tmdb.org/t/p/w500"


def _is_v4(token: str) -> bool:
    """判断是 v4 Read Access Token（JWT）还是 v3 API Key。"""
    t = (token or "").strip()
    return t.startswith("eyJ") and t.count(".") >= 2


def _detect_proxies() -> dict | None:
    """探测代理：优先环境变量，Windows 上再读系统代理设置。返回 requests 用的 proxies dict。"""
    # 1. 显式环境变量（启动器/命令行注入的优先）
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        v = os.environ.get(key)
        if v:
            return {"http": v, "https": v}
    # 2. Windows 系统代理（IE / 设置 → 网络 → 代理）
    if sys.platform.startswith("win"):
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            ) as k:
                enabled, _ = winreg.QueryValueEx(k, "ProxyEnable")
                if enabled:
                    server, _ = winreg.QueryValueEx(k, "ProxyServer")
                    if server:
                        if "=" in server:  # 形如 http=127.0.0.1:7890;https=127.0.0.1:7890
                            proxies: dict[str, str] = {}
                            for part in server.split(";"):
                                if "=" in part:
                                    proto, addr = part.split("=", 1)
                                    proxies[proto.strip().lower()] = addr.strip()
                            if proxies:
                                return proxies
                        return {"http": server, "https": server}
        except Exception:  # noqa: BLE001
            pass
    return None


def _get(token: str, path: str, params: dict | None = None,
         proxies: dict | None = None, log=print) -> dict:
    """GET TMDb 端点，返回 JSON。v4 走 Bearer 头，v3 走 api_key 参数。
    401 直接报错（Key 无效），429 退避重试，连接超时给出代理提示。"""
    params = params or {}
    headers = {"Accept": "application/json"}
    if _is_v4(token):
        headers["Authorization"] = f"Bearer {token.strip()}"
    else:
        params["api_key"] = token.strip()

    if proxies is None:
        proxies = _detect_proxies()
    req_kwargs = {"params": params, "headers": headers, "timeout": 30}
    if proxies:
        req_kwargs["proxies"] = proxies

    last = None
    for attempt in range(3):
        try:
            r = requests.get(f"{TMDB_V3}{path}", **req_kwargs)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ConnectTimeout) as e:
            raise RuntimeError(
                "无法连接 api.themoviedb.org（连接超时）。"
                "在中国大陆通常需要代理/VPN：请在启动器「代理」框填入代理地址"
                "（如 http://127.0.0.1:7890），或确认系统代理已开启后重试。"
                f"原始错误：{e}"
            ) from e
        if r.status_code == 401:
            raise RuntimeError("TMDb 返回 401：API Key / Token 无效，请在启动器面板重新填写。")
        if r.status_code == 429:  # 限流，退避后重试
            time.sleep(2 * (attempt + 1))
            last = r
            continue
        r.raise_for_status()
        return r.json()
    if last is not None:
        last.raise_for_status()
    return {}


def _download_poster_bytes(poster_path: str | None,
                            proxies: dict | None = None, log=print) -> bytes | None:
    """下载海报字节（不落盘，直接返回 bytes 供存库）。失败返回 None。"""
    if not poster_path:
        return None
    url = f"{IMG_BASE}{poster_path}"
    try:
        r = requests.get(url, timeout=30, proxies=proxies or _detect_proxies())
        if r.status_code == 200 and r.content:
            return r.content
        log(f"[图片] 下载失败（HTTP {r.status_code}）：{url}")
    except Exception as e:  # noqa: BLE001
        log(f"[图片] 下载异常：{e}")
    return None


def transform_movie(detail: dict, credits: dict | None = None,
                    external: dict | None = None) -> dict | None:
    """把 TMDb movie details 映射成 movies 表 dict。纯函数，便于单测。"""
    if not detail:
        return None
    # 类型：按 TMDb genre_id 映射成项目统一中文名（与 genres 类型表完全一致），逗号分隔
    genre_names = [genre_name(g.get("id")) or g.get("name") for g in detail.get("genres", [])]
    genre_names = [n for n in genre_names if n]
    genres = ", ".join(genre_names)
    countries = " / ".join(c.get("name", "") for c in detail.get("production_countries", []) if c.get("name"))
    languages = " / ".join(l.get("name", "") for l in detail.get("spoken_languages", []) if l.get("name"))
    runtime = detail.get("runtime")
    runtime = f"{runtime}分钟" if isinstance(runtime, int) else None
    release = detail.get("release_date") or ""
    year = int(release[:4]) if release[:4].isdigit() else None
    directors = writers = casts = None
    if credits:
        directors = " / ".join(p.get("name", "") for p in credits.get("crew", [])
                               if p.get("job") == "Director" and p.get("name"))
        writers = " / ".join(p.get("name", "") for p in credits.get("crew", [])
                             if p.get("job") in ("Writer", "Screenplay") and p.get("name"))
        casts = " / ".join(p.get("name", "") for p in credits.get("cast", [])[:8] if p.get("name"))
    imdb = (external or {}).get("imdb_id") or detail.get("imdb_id")
    poster = detail.get("poster_path")
    # 默认用远程 CDN；run() 里会尝试下载到本地并覆盖为 /static 路径
    poster_url = f"{IMG_BASE}{poster}" if poster else None
    return {
        "tmdb_id": str(detail.get("id")),
        "title": detail.get("title") or detail.get("original_title") or "未知",
        "original_title": detail.get("original_title"),
        "year": year,
        "directors": directors,
        "writers": writers,
        "casts": casts,
        "genres": genres,
        "country": countries,
        "language": languages,
        "release_date": release or None,
        "runtime": runtime,
        "aka": None,
        "imdb": imdb,
        "rating": detail.get("vote_average"),
        "rating_count": detail.get("vote_count"),
        "popularity": detail.get("popularity"),
        "summary": detail.get("overview"),
        "source": "tmdb",
        "tagline": detail.get("tagline"),
        "poster_url": poster_url,
    }


def transform_review(rv: dict, movie_id: int) -> dict | None:
    """把 TMDb review 映射成 reviews 表 dict。纯函数，便于单测。"""
    if not rv:
        return None
    content = rv.get("content") or ""
    rating = (rv.get("author_details") or {}).get("rating")
    rating_val = None
    if isinstance(rating, (int, float)) and rating:
        # TMDb 评分 1-10，本表 rating 为 1-5，按比例折算
        rating_val = max(1, min(5, round(rating / 2)))
    pub = rv.get("created_at")
    publish_date = None
    if pub:
        try:
            publish_date = datetime.strptime(pub[:19], "%Y-%m-%dT%H:%M:%S")
        except (ValueError, TypeError):
            pass
    return {
        "tmdb_review_id": str(rv.get("id")),
        "movie_id": movie_id,
        "reviewer_id": None,
        "title": f"{(rv.get('author') or '匿名')} 的影评",
        "rating": rating_val,
        "rating_label": None,
        "summary": content[:200],
        "content": content,
        "useful_count": 0,
        "publish_date": publish_date,
    }


def transform_cast(credits: dict | None, limit: int = 12) -> list[dict]:
    """把 TMDb credits.cast 映射成 movie_cast 行。纯函数，便于单测。
    取前 limit 位演员（TMDb 已按重要度排序），保留 饰演角色 与 头像路径。"""
    out = []
    for p in (credits or {}).get("cast", [])[:limit]:
        name = p.get("name")
        if not name:
            continue
        out.append({
            "tmdb_person_id": str(p.get("id")) if p.get("id") else None,
            "name": name,
            "character": (p.get("character") or "").strip() or None,
            "order": p.get("order", 0) or 0,
            "profile_path": p.get("profile_path"),
        })
    return out


def _fetch_candidates(api_key, count, mode, genre_id, sort_by, year, min_votes,
                      search, language, proxies, log):
    """按 模式/类型/排序/年份/最少投票/关键词 拉取候选电影列表（自动分页凑够 count 部）。

    对应 TMDb 端点：
      - search    → /search/movie?query=        （单一关键词条件）
      - top_rated → /movie/top_rated            （高分榜）
      - discover  → /discover/movie?with_genres=&sort_by=&primary_release_year=&vote_count.gte=
                    （多条件组合：类型 + 排序 + 年份 + 最少评分人数）
      - popular   → /movie/popular              （热度榜，默认）
    返回 list[dict]（TMDb movie 摘要，含 id/title/release_date 等）。
    """
    candidates = []
    page = 1
    max_pages = 20  # TMDb 单端点最多 20 页 × 20 = 400 条
    while len(candidates) < count and page <= max_pages:
        if search:
            data = _get(api_key, "/search/movie",
                        {"language": language, "query": search, "page": page, "include_adult": "false"},
                        proxies, log)
        elif mode == "top_rated":
            data = _get(api_key, "/movie/top_rated", {"language": language, "page": page}, proxies, log)
        elif mode == "discover":
            params = {"language": language, "page": page,
                      "sort_by": sort_by or "popularity.desc"}
            if genre_id:
                params["with_genres"] = genre_id
            if year:
                params["primary_release_year"] = year
            if min_votes:
                params["vote_count.gte"] = min_votes
            data = _get(api_key, "/discover/movie", params, proxies, log)
        else:  # popular
            data = _get(api_key, "/movie/popular", {"language": language, "page": page}, proxies, log)
        results = data.get("results", [])
        if not results:
            break
        candidates.extend(results)
        page += 1
    return candidates[:count]


def run(api_key: str, count: int = 20, language: str = "zh-CN",
        search: str | None = None,
        mode: str = "popular",
        genre_id: int | None = None,
        sort_by: str = "popularity.desc",
        year: int | None = None,
        min_votes: int | None = None,
        only_new: bool = False,
        proxies: dict | None = None, log=print):
    """拉取电影（支持 热门/高分/按类型发现/关键词搜索 + 多条件筛选 + 仅爬新数据）+
    各自影评入库，并下载海报到数据库 BLOB。

    参数：
      mode      : popular(热门) | top_rated(高分) | discover(按类型发现)
      genre_id  : TMDb 类型 id（discover 模式生效；None=全部类型）
      sort_by   : discover 排序，如 popularity.desc / vote_average.desc / primary_release_date.desc
      year      : 上映年份（discover 生效）
      min_votes : 最少评分人数（discover 生效，过滤小众影片）
      only_new  : True 时跳过库中已存在的 tmdb_id（不重复拉取/覆盖）
    返回 {movies, reviews, skipped}。
    """
    if proxies is None:
        proxies = _detect_proxies()
        if proxies:
            log(f"[网络] 已启用代理：{proxies.get('https') or proxies.get('http')}")

    db = SessionLocal()
    try:
        candidates = _fetch_candidates(api_key, count, mode, genre_id, sort_by, year,
                                      min_votes, search, language, proxies, log)
        total = len(candidates)
        if total == 0:
            log("[完成] 没有拿到任何电影，请检查：Key 是否有效 / 网络能否访问 api.themoviedb.org"
                "（是否需要代理）/ 条件是否过窄（如类型+年份组合无结果）。")
            return {"movies": 0, "reviews": 0, "skipped": 0}

        saved_movies = saved_reviews = skipped = 0
        for i, m in enumerate(candidates, 1):
            mid = m.get("id")
            # 仅爬新数据：库中已有该 tmdb_id 则跳过（省一次详情请求 + 海报下载）
            if only_new and crud.get_movie_by_tmdb_id(db, str(mid)):
                log(f"[跳过] tmdb_id={mid} 已存在，仅爬新数据模式下跳过")
                skipped += 1
                continue
            log(f"[STEP] {i}/{total} 抓取电影：{m.get('title')} ({str(m.get('release_date', ''))[:4]})")
            # 一次调用带回 详情 + 导演/演员(credits) + IMDb(external_ids) + 影评(reviews)
            detail = _get(api_key, f"/movie/{mid}",
                          {"language": language,
                           "append_to_response": "credits,external_ids,reviews"},
                          proxies, log)
            movie_data = transform_movie(detail, detail.get("credits"), detail.get("external_ids"))
            movie = crud.create_movie(db, movie_data)
            # 演员表（结构化：演员 + 饰演角色 + 头像路径），整批替换保证与 TMDb 一致
            cast_list = transform_cast(detail.get("credits"))
            if cast_list:
                crud.replace_movie_cast(db, movie.id, cast_list)
                log(f"[演员] 已写入 {len(cast_list)} 位演员")
            # 海报：下载字节直接存库（不放文件夹），访问地址指向 /api/movies/{id}/poster
            blob = _download_poster_bytes(detail.get("poster_path"), proxies, log)
            if blob:
                movie.poster = blob
                movie.poster_url = f"/api/movies/{movie.id}/poster"
                db.commit()
                log(f"[图片] 海报已存入数据库（{len(blob)} 字节）")
            else:
                log("[图片] 海报下载失败，poster_url 保留为 TMDb 远程链接（需浏览器能访问 image.tmdb.org）")
            # 影评（前 5 条，来自上面合并返回的 reviews，省一次请求）
            reviews_block = detail.get("reviews") or {}
            for rv in (reviews_block.get("results") or [])[:5]:
                rd = transform_review(rv, movie.id)
                author = rv.get("author") or "匿名"
                reviewer = crud.get_or_create_reviewer(db, name=author)
                rd["reviewer_id"] = reviewer.id
                crud.create_review(db, rd)
                saved_reviews += 1
            saved_movies += 1
            time.sleep(0.3)  # 友好限速，避免触发 TMDb 限流
        log(f"[完成] 共入库 {saved_movies} 部电影 / {saved_reviews} 条影评"
            + (f"（跳过已存在 {skipped} 部）" if skipped else "")
            + "（海报已存入数据库 BLOB，经 /api/movies/{id}/poster 回传）")
        return {"movies": saved_movies, "reviews": saved_reviews, "skipped": skipped}
    finally:
        db.close()


if __name__ == "__main__":
    key = os.environ.get("TMDB_API_KEY")
    if not key:
        print("[错误] 未设置 TMDB_API_KEY 环境变量（在启动器面板填写，或 export TMDB_API_KEY=xxx）")
        sys.exit(1)
    try:
        count = int(os.environ.get("TMDB_COUNT", "20"))
    except ValueError:
        count = 20
    search = os.environ.get("TMDB_SEARCH") or None
    language = os.environ.get("TMDB_LANGUAGE", "zh-CN")
    mode = os.environ.get("TMDB_MODE", "popular")
    genre_id = os.environ.get("TMDB_GENRE_ID")
    genre_id = int(genre_id) if genre_id and genre_id.isdigit() else None
    sort_by = os.environ.get("TMDB_SORT_BY", "popularity.desc")
    year = os.environ.get("TMDB_YEAR")
    year = int(year) if year and year.isdigit() else None
    min_votes = os.environ.get("TMDB_MIN_VOTES")
    min_votes = int(min_votes) if min_votes and min_votes.isdigit() else None
    only_new = os.environ.get("TMDB_ONLY_NEW", "").lower() in ("1", "true", "yes")
    try:
        run(key, count=count, language=language, search=search,
            mode=mode, genre_id=genre_id, sort_by=sort_by,
            year=year, min_votes=min_votes, only_new=only_new)
    except Exception as e:  # noqa: BLE001
        print(f"[错误] 拉取失败：{e}")
        sys.exit(1)
