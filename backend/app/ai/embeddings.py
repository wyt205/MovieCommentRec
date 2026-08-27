"""嵌入模型封装：把文本变成向量。

统一走 OpenAI 兼容协议（与 GLM 用同一套 client），因此星火 MaaS 的
Qwen3-Embedding-8B 只需改 base_url / model / api_key 即可，无需新依赖。

实测：
- 端点 POST https://maas-api.cn-huabei-1.xf-yun.com/v2/embeddings
- 鉴权 Authorization: Bearer <完整 key（含冒号）>
- 返回 768 维向量，支持批量 input
"""

import time
import concurrent.futures

from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


def get_embedder() -> OpenAIEmbeddings:
    """返回一个 OpenAIEmbeddings 实例，指向星火 MaaS 嵌入端点。

    防御性加上 request_timeout / max_retries（老版本 langchain_openai 不支持这些
    kwarg 时自动降级），并配合 _embed_with_retry 的墙钟超时，杜绝「API 卡死 →
    整条建库进程永久挂起、数据库永远写不进去」的故障。
    """
    if not settings.embedding_api_key:
        raise RuntimeError(
            "未配置 EMBEDDING_API_KEY，RAG 语义检索未启用。请在 .env 中设置 EMBEDDING_API_KEY。"
        )
    kwargs = dict(
        model=settings.embedding_model or "xop3qwen8bembedding",
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url or "https://maas-api.cn-huabei-1.xf-yun.com/v2",
    )
    # 尽量让底层 HTTP 客户端自带超时/重试；失败则降级为不带这两个参数的构造
    try:
        kwargs["request_timeout"] = 30   # 单次 HTTP 请求超时（秒）
        kwargs["max_retries"] = 2
        return OpenAIEmbeddings(**kwargs)
    except TypeError:
        kwargs.pop("request_timeout", None)
        kwargs.pop("max_retries", None)
        return OpenAIEmbeddings(**kwargs)


def _embed_with_retry(embedder: "OpenAIEmbeddings", texts: list[str],
                       attempts: int = 3, base_delay: float = 2.0,
                       per_call_timeout: float = 45.0) -> list[list[float]]:
    """调用嵌入接口：退避重试 + 单次调用墙钟超时，杜绝永久卡死。

    关键加固：即使底层 openai 客户端无视 request_timeout（如卡在坏代理/半开连接），
    也用 ThreadPoolExecutor 的墙钟超时在 per_call_timeout 秒后强制放弃该次调用，
    转为重试；重试耗尽则抛出**清晰可读**的错误，而不是让整条建库进程无限挂起。

    鉴权类错误（401/403）快速失败、不空耗重试；其余异常按指数退避重试。
    """
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(embedder.embed_documents, texts)
                return fut.result(timeout=per_call_timeout)
        except concurrent.futures.TimeoutError:
            last_err = TimeoutError(
                f"嵌入接口单次调用超过 {per_call_timeout}s 未返回（疑似网络/代理卡死）")
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e).lower()
            # 鉴权失败：重试无意义，直接抛出
            if "401" in msg or "403" in msg or "unauthorized" in msg or "authentication" in msg:
                raise
            if i == attempts - 1:
                break
            time.sleep(base_delay * (2 ** i))
    raise RuntimeError(f"嵌入接口调用失败（已重试 {attempts} 次）：{last_err}") from last_err


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量将文本转为向量（用于建库/文档侧，不附加指令）。

    输入文本列表，返回同长度的向量列表。
    """
    if not texts:
        return []
    return _embed_with_retry(get_embedder(), texts)


# 检索场景下，查询(query)需与文档(document)落在同一语义子空间。
# 星火 MaaS 的 Qwen3-Embedding 端点未实现 task 参数，故用指令前缀对齐。
# 实测中文前缀对短查询对齐效果最好（英文前缀次之）。
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def embed_query(text: str) -> list[float]:
    """将单条查询转为向量（检索侧，附加查询指令前缀以对齐文档向量）。"""
    return _embed_with_retry(get_embedder(), [QUERY_INSTRUCTION + text])[0]
