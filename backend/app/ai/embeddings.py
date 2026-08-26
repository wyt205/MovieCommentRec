"""嵌入模型封装：把文本变成向量。

统一走 OpenAI 兼容协议（与 GLM 用同一套 client），因此星火 MaaS 的
Qwen3-Embedding-8B 只需改 base_url / model / api_key 即可，无需新依赖。

实测：
- 端点 POST https://maas-api.cn-huabei-1.xf-yun.com/v2/embeddings
- 鉴权 Authorization: Bearer <完整 key（含冒号）>
- 返回 768 维向量，支持批量 input
"""

from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


def get_embedder() -> OpenAIEmbeddings:
    """返回一个 OpenAIEmbeddings 实例，指向星火 MaaS 嵌入端点。"""
    if not settings.embedding_api_key:
        raise RuntimeError(
            "未配置 EMBEDDING_API_KEY，RAG 语义检索未启用。请在 .env 中设置 EMBEDDING_API_KEY。"
        )
    return OpenAIEmbeddings(
        model=settings.embedding_model or "xop3qwen8bembedding",
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url or "https://maas-api.cn-huabei-1.xf-yun.com/v2",
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量将文本转为向量（用于建库/文档侧，不附加指令）。

    输入文本列表，返回同长度的向量列表。
    """
    if not texts:
        return []
    return get_embedder().embed_documents(texts)


# 检索场景下，查询(query)需与文档(document)落在同一语义子空间。
# 星火 MaaS 的 Qwen3-Embedding 端点未实现 task 参数，故用指令前缀对齐。
# 实测中文前缀对短查询对齐效果最好（英文前缀次之）。
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def embed_query(text: str) -> list[float]:
    """将单条查询转为向量（检索侧，附加查询指令前缀以对齐文档向量）。"""
    return get_embedder().embed_documents([QUERY_INSTRUCTION + text])[0]
