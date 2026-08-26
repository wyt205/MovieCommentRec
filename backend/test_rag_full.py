"""RAG 全链路测试：建表核查 + 建库(幂等) + 语义检索验证 + agent 工具验证。"""
from sqlalchemy import text

from app.core.config import settings
from app.db.database import engine, SessionLocal
from app import models
from app.ai import rag
from app.ai.tools import semantic_search_movies


def section(t):
    print("\n" + "=" * 60)
    print(t)
    print("=" * 60)


# 1) 权威核查：用与后端完全相同的连接，列出库里所有表
section("1) 数据库权威核查")
print("DATABASE_URL =", settings.database_url)
with engine.connect() as c:
    tables = [r[0] for r in c.execute(text("SHOW TABLES")).fetchall()]
    print("llm_pro 库中的表：", tables)
    print("movie_embeddings 是否存在：", "movie_embeddings" in tables)
    if "movie_embeddings" in tables:
        cols = [r[0] for r in c.execute(text("DESC movie_embeddings")).fetchall()]
        print("movie_embeddings 列：", cols)

# 2) 建库（幂等，已存在则覆盖更新）
section("2) 重建语义索引（build_movie_embeddings）")
n = rag.build_movie_embeddings()
print("已处理电影数 =", n)

db = SessionLocal()
print("movie_embeddings 行数 =", db.query(models.MovieEmbedding).count())
db.close()

# 3) 语义检索验证（之前被打断的补充查询）
section("3) 语义检索验证（semantic_search）")
queries = [
    "蜘蛛侠守护纽约的超级英雄",
    "两个角色互换身体的搞笑冒险动画",
    "鬼杀队与鬼战斗的日本动画",
    "讲亲情和解的温馨故事",
    "一部讲时间旅行的科幻片",
]
for q in queries:
    top = rag.semantic_search(q, top_k=3)
    print(f"\n查询：{q}")
    for score, m in top:
        print(f"   {score:.3f}  《{m.title}》({m.year}) 类型:{m.genres}")

# 4) agent 工具封装验证（与 agent 实际调用路径一致）
section("4) agent 工具 semantic_search_movies 验证")
print(semantic_search_movies.invoke({"query": "守护城市的超级英雄", "top_k": 3}))

print("\n全部测试完成。")
