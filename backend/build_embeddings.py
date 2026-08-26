"""为 movies 表建立语义向量索引（RAG 建库）。

用法（在 backend 目录下运行）：
    python build_embeddings.py

依赖 .env 中的 EMBEDDING_API_KEY 等配置。会遍历 movies 表，
调用星火 MaaS 嵌入接口，将每部电影的向量写入 movie_embeddings 表。
可重复运行（已存在的 movie_id 覆盖更新，换模型后重跑即可重建索引）。
"""

from app.ai.rag import build_movie_embeddings


def main():
    print("开始为电影建立语义向量索引……")
    n = build_movie_embeddings()
    print(f"完成：已为 {n} 部电影建立/更新语义向量。")


if __name__ == "__main__":
    main()
