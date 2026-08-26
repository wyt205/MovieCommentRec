# -*- coding: utf-8 -*-
"""
智影 数据库结构迁移（安全，不丢数据）
========================================
仅对现有表做 ALTER：补齐 TMDb 相关列（poster / source / tagline / popularity），
删除豆瓣专属列（star_5~star_1 / douban_id / douban_review_id / douban_url /
douban_uid）与短评表 short_comments，必要时自动建库、建 genres 类型表并播种。

与 init.sql / seed.py 的区别：
  - init.sql    ：会 DROP 全部表再重建 → 清空所有数据，别随便跑。
  - seed.py     ：drop_all + create_all → 同样清空数据后只插示例。
  - migrate.py  ：只改结构、不动数据，可随时重跑（幂等）。

用法（在 backend 目录下，确保 MySQL 已启动）：
    conda activate llm-pro
    python migrate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入 database 模块即触发其中的 _migrate_columns() + _ensure_database()
# （这两个函数在模块底部、import 时自动执行，对已存在的库做幂等 ALTER）
from sqlalchemy import inspect
from app.db.database import engine, settings, Base, SessionLocal
import app.models  # 确保模型（含 Genre）被加载，create_all 才能建 genres 表
from app.core import genres as genres_catalog


def _ensure_genres():
    """确保 genres 类型表存在并播种 TMDb 官方 19 类（幂等）。"""
    try:
        Base.metadata.create_all(bind=engine)  # 建任何缺失的表（含 genres）
        with SessionLocal() as db:
            if db.query(app.models.Genre).count() == 0:
                for g in genres_catalog.TMDB_GENRES:
                    db.add(app.models.Genre(tmdb_id=g["id"], name=g["name"]))
                db.commit()
                print(f"[迁移] 已播种 {len(genres_catalog.TMDB_GENRES)} 个电影分类")
            else:
                print("[迁移] genres 分类目录已存在，跳过播种")
    except Exception as e:  # noqa: BLE001
        print(f"[迁移] genres 表处理跳过（请确认 MySQL 已启动）：{e}")


def main():
    target = settings.database_url.split("@")[-1]
    print(f"[迁移] 目标库：{target}")
    try:
        insp = inspect(engine)
        cols = {c["name"] for c in insp.get_columns("movies")}
    except Exception as e:  # noqa: BLE001
        print(f"[迁移] 无法连接 MySQL，迁移未执行：{e}")
        print("        请确认：① MySQL 服务已启动  ② backend/.env 的 DATABASE_URL 账号密码正确")
        sys.exit(1)

    new_cols = ("poster", "source", "tagline", "popularity")
    old_cols = ("star_5", "star_4", "star_3", "star_2", "star_1")
    douban_cols = ("douban_id",)
    added = [c for c in new_cols if c in cols]
    still_old = [c for c in old_cols if c in cols]
    still_douban = [c for c in douban_cols if c in cols]

    print("[迁移] 当前 movies 列：", ", ".join(sorted(cols)))
    print(f"[迁移] 新列已就位：{added or '无'}")
    print(f"[迁移] 旧 star_* 残留：{still_old or '无'}")
    print(f"[迁移] 豆瓣列残留（应在导入时自动删除）：{still_douban or '无'}")

    if set(new_cols).issubset(cols) and not still_old:
        print("[迁移] movies 结构已是最新。")
    else:
        print("[迁移] 若仍缺列，请确认 MySQL 连接正常后重跑本脚本，或直接重启后端触发自动迁移。")
    print("[迁移] 注：douban_* 列与 short_comments 表在导入本模块时已自动 DROP（纯 TMDb 化）。")

    _ensure_genres()


if __name__ == "__main__":
    main()
