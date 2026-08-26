# -*- coding: utf-8 -*-
"""
智影 示例数据种子脚本
================================
用法（在 backend 目录下，且已建好 backend/.env）：
    conda activate llm-pro
    python seed.py

行为：
  1. 确保数据库 llm_pro 存在
  2. 删除并重建全部表（drop_all + create_all，基于 ORM 模型）
  3. 插入示例：2 部电影 / 3 位评论者 / 3 条 TMDb 风格影评
连接串来自 app.core.config（即 backend/.env 的 DATABASE_URL）。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from app.models import Base
from app.core.config import settings

URL = settings.database_url
# 用于 "CREATE DATABASE" 的连接（不带具体库名）
BASE = URL.split("/", 3)[0] + "//" + URL.split("//", 1)[1].split("/", 1)[0] + "/"


def main():
    print("[*] 使用连接串:", URL)
    e0 = create_engine(BASE)
    with e0.connect() as c:
        c.execute(text(
            "CREATE DATABASE IF NOT EXISTS `llm_pro` "
            "DEFAULT CHARACTER SET utf8mb4 DEFAULT COLLATE utf8mb4_unicode_ci"
        ))
        c.commit()
    print("[1] 数据库 llm_pro 就绪")

    engine = create_engine(URL, pool_pre_ping=True)
    with engine.connect() as c:
        c.execute(text("SET FOREIGN_KEY_CHECKS=0"))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with engine.connect() as c:
        c.execute(text("SET FOREIGN_KEY_CHECKS=1"))
    print("[2] 表已重建（movies/reviewers/reviews，纯 TMDb 结构）")

    with engine.connect() as conn:
        # TMDb 真实 id：肖申克的救赎=278，星际穿越=157336
        movies = [
            dict(tmdb_id="278", title="肖申克的救赎",
                 original_title="The Shawshank Redemption", year=1994,
                 directors="弗兰克·德拉邦特", writers="弗兰克·德拉邦特",
                 casts="蒂姆·罗宾斯,摩根·弗里曼,鲍勃·冈顿", genres="犯罪,剧情",
                 country="美国", language="英语",
                 release_date="1994-09-10", runtime="142分钟",
                 aka="月黑高飞(港),刺激1995(台)", imdb="tt0111161",
                 rating=9.3, rating_count=2700000, popularity=95.5,
                 source="tmdb",
                 summary="蒙冤入狱的银行家安迪用近二十年耐心与智慧完成自我救赎，"
                         "重燃狱友瑞德对自由的渴望。",
                 poster_url=None),
            dict(tmdb_id="157336", title="星际穿越",
                 original_title="Interstellar", year=2014,
                 directors="克里斯托弗·诺兰", writers="克里斯托弗·诺兰,乔纳森·诺兰",
                 casts="马修·麦康纳,安妮·海瑟薇,杰西卡·查斯坦", genres="剧情,科幻,冒险",
                 country="美国,英国", language="英语",
                 release_date="2014-11-12", runtime="169分钟",
                 aka="星际启示录(港),星际效应(台)", imdb="tt0816692",
                 rating=8.6, rating_count=2100000, popularity=118.2,
                 source="tmdb",
                 summary="近未来地球环境崩溃，前宇航员库珀穿越虫洞寻找新家园，"
                         "在相对论时间尺度下父女之爱跨越星际与岁月。",
                 poster_url=None),
        ]
        conn.execute(text(
            "INSERT INTO movies (tmdb_id,title,original_title,year,directors,writers,casts,"
            "genres,country,language,release_date,runtime,aka,imdb,rating,rating_count,popularity,"
            "source,summary,poster_url) "
            "VALUES (:tmdb_id,:title,:original_title,:year,:directors,:writers,:casts,"
            ":genres,:country,:language,:release_date,:runtime,:aka,:imdb,:rating,:rating_count,"
            ":popularity,:source,:summary,:poster_url)"), movies)
        mid = {d: i for i, d in conn.execute(
            text("SELECT id,tmdb_id FROM movies WHERE tmdb_id IN (:a,:b)"),
            {"a": "278", "b": "157336"}).all()}
        print("    movie ids:", mid)

        # 演员表（示例，profile_path 留空 → 前端用字母头像兜底，离线也能看）
        cast = [
            dict(movie_id=mid["278"], name="蒂姆·罗宾斯", character="安迪·杜佛兰", order=0),
            dict(movie_id=mid["278"], name="摩根·弗里曼", character="艾利斯·“瑞德”·瑞丁", order=1),
            dict(movie_id=mid["278"], name="鲍勃·冈顿", character="山姆·诺顿监狱长", order=2),
            dict(movie_id=mid["278"], name="威廉姆·赛德勒", character="海伍德", order=3),
            dict(movie_id=mid["157336"], name="马修·麦康纳", character="约瑟夫·库珀", order=0),
            dict(movie_id=mid["157336"], name="安妮·海瑟薇", character="艾米莉亚·布兰德", order=1),
            dict(movie_id=mid["157336"], name="杰西卡·查斯坦", character="成年墨菲", order=2),
            dict(movie_id=mid["157336"], name="迈克尔·凯恩", character="布兰德教授", order=3),
        ]
        conn.execute(text(
            "INSERT INTO movie_cast (movie_id,name,character,`order`) "
            "VALUES (:movie_id,:name,:character,:order)"), cast)
        print("    演员行:", len(cast))

        reviewers = [
            dict(name="影评人老张",
                 avatar_url="https://placehold.co/80x80?text=Z", location="北京",
                 signature="电影是时间的容器"),
            dict(name="诺兰铁粉",
                 avatar_url="https://placehold.co/80x80?text=N", location="上海",
                 signature="IMAX 信徒"),
            dict(name="路人甲",
                 avatar_url="https://placehold.co/80x80?text=A", location="广州",
                 signature="随便看看"),
        ]
        conn.execute(text(
            "INSERT INTO reviewers (name,avatar_url,location,signature) "
            "VALUES (:name,:avatar_url,:location,:signature)"), reviewers)
        rid = {u: i for i, u in conn.execute(
            text("SELECT id,name FROM reviewers WHERE name IN (:a,:b,:c)"),
            {"a": "影评人老张", "b": "诺兰铁粉", "c": "路人甲"}).all()}
        print("    reviewer ids:", rid)

        reviews = [
            dict(tmdb_review_id="rev_shawshank_1", movie_id=mid["278"],
                 reviewer_id=rid["影评人老张"], title="希望是一件好事，也许是最好的事",
                 rating=5, rating_label="力荐",
                 summary="有些电影看完只会讨论剧情，这部看完会想重新活一次。",
                 content="安迪用一把小锤子挖了十九年，真正挖穿的是体制化的高墙。"
                         "当安迪在雨中张开双臂，那不是越狱的终点，而是人之为人的确认。",
                 useful_count=15234, comments_count=482, views=99120,
                 publish_date="2018-05-02 21:13:00"),
            dict(tmdb_review_id="rev_interstellar_1", movie_id=mid["157336"],
                 reviewer_id=rid["诺兰铁粉"], title="诺兰把相对论拍成了情书",
                 rating=5, rating_label="力荐",
                 summary="看不懂物理没关系，看得懂父女就够了。",
                 content="黑洞的引力把时间拉成丝，一小时等于七年。库珀看着孩子们长大，"
                         "自己却几乎没变老。汉斯·季默的管风琴一响，眼泪就下来了。",
                 useful_count=22105, comments_count=673, views=120330,
                 publish_date="2019-11-20 10:02:00"),
            dict(tmdb_review_id="rev_interstellar_2", movie_id=mid["157336"],
                 reviewer_id=rid["路人甲"], title="视听很爽，但有点长",
                 rating=4, rating_label="推荐",
                 summary="三小时坐得住，就是结尾稍强行。",
                 content="画面和音效值回票价，整体仍是近年最好的硬科幻之一。",
                 useful_count=8830, comments_count=210, views=55210,
                 publish_date="2020-01-15 19:40:00"),
        ]
        conn.execute(text(
            "INSERT INTO reviews (tmdb_review_id,movie_id,reviewer_id,source,title,rating,rating_label,"
            "summary,content,useful_count,comments_count,views,publish_date) "
            "VALUES (:tmdb_review_id,:movie_id,:reviewer_id,'tmdb',:title,:rating,:rating_label,"
            ":summary,:content,:useful_count,:comments_count,:views,:publish_date)"),
            reviews)
        conn.commit()
        print("[3] 示例数据插入完成：2 部电影 / 3 位评论者 / 3 条影评")

        for t in ("movies", "reviewers", "reviews"):
            print("    表 %s: %d 行" % (t, conn.execute(
                text("SELECT COUNT(*) FROM " + t)).scalar()))


if __name__ == "__main__":
    main()
