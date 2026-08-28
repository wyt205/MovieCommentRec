# backend（FastAPI）

电影推荐 Agent 的后端：REST API + MySQL 存储 + LangChain Function Calling 自主 Agent
+ TMDb 数据管道 + 语义向量检索（RAG）。由根目录 `docker-compose.yml` 编排为
`zhiying-backend` 服务。

## 目录结构
```
backend/
├── app/
│   ├── main.py            # FastAPI 入口 + CORS(*) + 自动建表 + 挂载路由 + /health
│   ├── core/
│   │   ├── config.py      # 读取 .env（DATABASE_URL / LLM_* / EMBEDDING_*）
│   │   └── genres.py      # 类型词表
│   ├── db/database.py     # SQLAlchemy engine / SessionLocal / Base
│   ├── models.py          # ORM 模型（9 张表）
│   ├── schemas.py         # Pydantic 响应模型
│   ├── crud.py            # 增删查（多类型 AND 过滤 / 模糊匹配 / upsert）
│   ├── runtime_flags.py   # 运行时开关（自主 Agent 模式）
│   ├── api/               # 路由（统一前缀 /api）
│   │   ├── movies.py      # 列表 / 详情 / 海报 / 演员头像 / stats
│   │   ├── reviews.py     # 影评 CRUD
│   │   ├── genres.py      # 类型列表
│   │   ├── agent.py       # /mode（切自主模式）/ chat / chat/stream（SSE）
│   │   ├── sessions.py    # 会话列表 / 新建 / 续聊 / 删除
│   │   └── admin.py       # 调用日志 / traces / stats / 评测运行
│   ├── ai/                # Agent 核心
│   │   ├── agent.py       # 护栏自主 agent（bind_tools + 护栏 + 确定性安全网）
│   │   ├── tools.py       # @tool 工具（find_movies / get_movie_info_by_name / semantic_search_movies ...）
│   │   ├── rag.py         # 混合检索（向量 + 关键词）+ 置信度阈值
│   │   └── embeddings.py  # 星火 Qwen3-Embedding-8B 客户端
│   └── crawler/
│       └── tmdb.py        # TMDb 爬虫（电影 / 演员 / 影评 / 海报 BLOB）
├── build_embeddings.py    # 建 / 重建电影语义向量库（写 movie_embeddings）
├── seed.py                # 最小示例数据
├── eval_cases.json        # 评测用例（22 条，11 类）
├── run_eval.py            # 评测运行器 → eval_result.json
├── requirements.txt
├── .env.example
└── Dockerfile
```

## 快速开始（本地，非 Docker）
```bash
# 1. 建库（见 ../sql/init.sql，后端启动时也会自动建表）
mysql -u root -p < ../sql/init.sql
# 2. 安装依赖（建议虚拟环境）
pip install -r requirements.txt
# 3. 配置
cp .env.example .env        # 填入 MySQL 账号密码 + LLM / EMBEDDING key
# 4. 启动
uvicorn app.main:app --reload --port 8000
```
接口文档：http://localhost:8000/docs

## 数据表（MySQL）
建表由 backend 启动时自动执行，DDL 见 `../sql/init.sql`。
| 表 | 说明 |
|----|------|
| `movies` | 电影（片名 / 年份 / 导演 / 演员 / 类型 / 国家 / 评分 / 简介 / **海报 LONGBLOB**） |
| `movie_cast` | 演员表（片名 ↔ 演员） |
| `movie_embeddings` | 电影语义向量（星火 Qwen3-Embedding-8B，768 维） |
| `reviewers` | 影评人 |
| `genres` | 类型词表 |
| `reviews` | 影评 |
| `agent_traces` | 对话可观测埋点（意图 / 工具链 / 护栏 / 缓存 / 耗时） |
| `chat_sessions` | 会话记录（DB-backed，可继续 / 删除） |
| `user_preferences` | 用户长期偏好（注入 system prompt） |

> 海报 / 演员头像存 MySQL BLOB，经 `/api/movies/{id}/poster`、`/api/movies/cast-photo`
> 回传，前端 `<img>` 直连，**不依赖本地文件夹**。

## API 一览（前缀 /api）
- `GET  /api/movies?keyword=&genre=&skip=&limit=` 电影列表（类型 AND 过滤 + 模糊）
- `GET  /api/movies/{id}` 电影详情
- `GET  /api/movies/{id}/poster` 海报（库内 BLOB 回传）
- `GET  /api/movies/cast-photo` 演员头像
- `GET  /api/movies/stats` 统计
- `GET  /api/reviews?movie_id=&skip=&limit=` 影评列表
- `POST /api/reviews` 新增影评
- `GET  /api/genres` 类型列表
- `POST /api/agent/chat`、`POST /api/agent/chat/stream`（SSE 流式）、`GET|POST /api/agent/mode`
- `GET|POST|DELETE /api/sessions[/{id}]` 会话管理
- 管理端：`GET /api/admin/traces`、`/api/admin/stats`、`/api/admin/eval`、`POST /api/admin/eval/run`

## 语义向量 & 爬虫
```bash
# 建 / 重建向量库（走国内星火 embedding，不需代理）
python build_embeddings.py

# TMDb 爬虫（环境变量驱动；本机出网需代理时加 -e HTTPS_PROXY=http://<宿主IP>:端口）
python -m app.crawler.tmdb            # 默认 20 部 popular
# 常用开关：
#   TMDB_COUNT       数量
#   TMDB_MODE        popular | top_rated | discover | search
#   TMDB_GENRE_ID    按类型抓
#   TMDB_ONLY_NEW    true 时跳过已入库影片（去重）
```

## Agent 架构（混合：规则路由为主 + Function Calling 兜底）
- **主路径（默认）**：确定性意图分类 → 规则抽参 → 调对应工具 → LLM 仅润色。可靠、零幻觉。
- **兜底路径（自主模式）**：`runtime_flags` 开启后，`llm.bind_tools` 真·Function Calling
  让模型自主选工具 + 护栏（未调工具则强制重试，失败回退确定性路由）。
- **抗幻觉**：答案 100% 锚定 DB；库内无此片 → 先查库确认、再如实说明 + 代码强制追加免责声明。

## Docker
由根目录 `docker-compose.yml` 编排；本目录 `Dockerfile` 基于 `python:3.11-slim` 构建
`zhiying-backend` 镜像（`.dockerignore` 已排除 `.env` / 缓存 / 陈旧海报）。
环境变量通过根 `.env`（`env_file`）注入，连库地址由 compose 覆盖为 `mysql` 服务名。
启动与停止见主目录 README「Docker 容器化部署」一节。
