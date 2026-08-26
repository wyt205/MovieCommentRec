from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # MySQL 连接串，见 .env.example
    database_url: str = "mysql+pymysql://root:password@localhost:3306/llm_pro?charset=utf8mb4"
    # TMDb 公开 API 凭证（v4 Read Access Token 或 v3 API Key），爬虫启动器/脚本会用到
    tmdb_api_key: str = ""

    # ---- 大模型（Agent 大脑）配置 ----
    # 模型待定时全部留空，Agent 路由会返回友好提示，不影响后端其他功能启动。
    # 支持任意 OpenAI 兼容协议：GLM-4-Flash / 讯飞 Ultra / GPT 等。
    llm_api_key: str = ""          # 模型 API Key
    llm_base_url: str = ""         # OpenAI 兼容端点，如 GLM-4-Flash: https://open.bigmodel.cn/api/paas/v4
    llm_model: str = "glm-4-flash-250414"  # 模型名

    # ---- 嵌入模型（RAG 语义检索）----
    # 星火 MaaS 免费 Qwen3-Embedding-8B，OpenAI 兼容 /v2/embeddings，返回 768 维向量。
    # 注意 base_url 写到 /v2，OpenAI SDK 会自动补成 /v2/embeddings。
    embedding_api_key: str = ""    # 星火 MaaS API Key（含冒号整串，如 xxxx:yyyy）
    embedding_base_url: str = "https://maas-api.cn-huabei-1.xf-yun.com/v2"
    embedding_model: str = "xop3qwen8bembedding"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
