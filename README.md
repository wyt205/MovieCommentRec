# 智影 (CineSage) · 电影推荐 Agent + 影评网站 + 可观测管理端

一个**前后端分离 + 已容器化**的电影应用：后端 FastAPI + MySQL，前端 Vue 3 + Vite，并接入了
**LangChain Function Calling 自主 Agent**（推荐 / 检索 / 闲聊 / 身份问答），外加一个独立的
**可观测管理端**（对话日志 / 缓存命中 / 评测看板）。已通过 `docker-compose` 一键编排
`mysql` / `backend` / `web` 三服务，可本地或云端部署。

> 项目定位：既是一个能跑的电影推荐产品原型，也是一次完整的 AI 应用工程实践（RAG + Agent + 容器化）。

## 目录结构
```
llm-pro/
├── docker-compose.yml       # 三服务编排：mysql / backend / web（容器名前缀 zhiying-*）
├── .env                     # Docker 用（根；含 LLM / 嵌入 / TMDb key 与 MySQL 配置，勿提交仓库）
├── docker流程.md            # 本地 Docker 入坑、排错与踩坑笔记
├── sql/init.sql             # MySQL 建库建表（纯 TMDb 体系）
├── launcher.py              # 本地 Windows 启动器（Tkinter GUI：前后端/爬虫/向量库/管理端 统一管理）
├── backend/                 # FastAPI 服务（详见 backend/README.md）
│   ├── app/
│   │   ├── main.py          # 入口 + CORS(*) + 自动建表 + 挂载路由 + /health
│   │   ├── core/{config,genres}.py
│   │   ├── db/database.py   # SQLAlchemy engine / SessionLocal / Base
│   │   ├── models.py        # ORM（movies / movie_cast / movie_embeddings / reviewers / genres / reviews / agent_traces / chat_sessions / user_preferences）
│   │   ├── schemas.py       # Pydantic 响应模型
│   │   ├── crud.py          # 增删查（多类型 AND 过滤 / 模糊匹配 / upsert）
│   │   ├── runtime_flags.py # 运行时开关（自主 Agent 模式）
│   │   ├── api/             # movies / reviews / genres / agent / admin / sessions
│   │   ├── ai/              # agent.py(护栏自主agent) / tools.py(7个@tool) / rag.py(混合检索) / embeddings.py(星火embedding)
│   │   └── crawler/tmdb.py  # TMDb 爬虫（电影 / 演员 / 影评 / 海报 BLOB）
│   ├── build_embeddings.py  # 建 / 重建电影语义向量库
│   ├── seed.py              # 最小示例数据
│   ├── eval_cases.json      # 评测用例（22 条，覆盖 11 类）
│   ├── run_eval.py          # 评测运行器 → eval_result.json
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/                # Vue 3 + Vite（C 端）+ Dockerfile + nginx.conf（详见 frontend/README.md）
│   └── src/{api,components,views,router}   # 含 AgentChat.vue 对话页
└── admin/                   # 独立管理端（纯静态 HTML/CSS/JS，随前端镜像一起打进 nginx）
    ├── index.html           # 四面板：Agent 对话 / 调用日志 / 缓存与统计 / 评测
    ├── styles.css
    └── app.js
```

## 技术栈
- **后端**：FastAPI + SQLAlchemy + MySQL + LangChain 1.x（`bind_tools`）
- **LLM（对话大脑）**：智谱 GLM-4-Flash（永久免费、原生 function calling，OpenAI 兼容）；本地用 `glm-4-flash-250414` 亦可
- **Embedding（语义检索）**：星火 MaaS Qwen3-Embedding-8B（768 维，OpenAI 兼容 `/v2/embeddings`）
- **前端**：Vue 3 + Vite（C 端）；原生 HTML/CSS/JS（管理端 `admin/`）
- **检索**：MySQL 内嵌向量 + Python 暴力余弦 + 关键词混合检索
- **部署**：Docker Compose（mysql + backend + web/nginx 三服务，Nginx 反代 `/api` 与 `/static`）

## 数据来源（TMDb）
- 电影 / 演员 / 影评来自 **TMDb**（`app/crawler/tmdb.py`），字段对应 `sql/init.sql`：片名 / 年份 / 导演 / 演员 / 类型 / 国家 / 上映日期 / 评分 / 评分人数 / 简介。
- 海报 / 演员头像以 **BLOB 存进 MySQL**（列类型 `LONGBLOB`），经 `/api/movies/{id}/poster`、`/api/movies/cast-photo` 回传前端，**不依赖本地文件夹**。
- 语义检索库：`build_embeddings.py` 切片 + 调星火 embedding 入库 `movie_embeddings`。

## 运行方式
### 方式一：Docker 容器化
见下方「Docker 容器化部署」一节。一条 `docker compose up -d --build` 起全套，任何装了 Docker 的机器都能复现。

### 方式二：本地启动器（Windows 调试）
```bash
conda activate llm-pro      # 依赖装在 conda 的 llm-pro 环境
python launcher.py
```
GUI 内可：一键启动前后端、重跑 TMDb 爬虫、建立语义向量库、起管理端（5180）。

### 方式三：手动
```bash
# 1. 建库
mysql -u root -p < sql/init.sql
# 2. 后端
cd backend && pip install -r requirements.txt && cp .env.example .env && uvicorn app.main:app --reload --port 8000
# 3. 前端
cd frontend && npm install && npm run dev          # http://localhost:5173
# 4. 管理端
cd admin && python -m http.server 5180            # http://localhost:5180
```

---

## Docker 容器化部署
三服务：`mysql`（数据卷持久化）/ `backend`（FastAPI）/ `web`（nginx 托管前端 + 管理端，并反代 `/api`、`/static` 到后端）。
compose 项目名默认 `llm-pro`（取目录名），容器名前缀 `zhiying-*`（mysql / backend / web），三者通过 `zhiying` 网络互联。

### 前置
- 安装 Docker Desktop（Windows 开启 WSL2 后端）。
- 避免权限 `denied`：`sudo usermod -aG docker $USER`（Linux/WSL），重开终端生效。

### 启动
```bash
# 1) 准备根 .env（填 LLM / 嵌入 / TMDb key；MYSQL/库名有默认值可不动）
cp backend/.env.example .env
# 2) 构建并后台启动三容器
docker compose up -d --build
# 3) 看状态（应三容器 Up，mysql healthy）
docker compose ps
```
访问入口：
| 网址 | 说明 |
|------|------|
| **http://localhost/** | 前端主页（nginx 托管） |
| **http://localhost/admin/** | 管理端（四面板：对话 / 日志 / 统计 / 评测） |
| http://localhost/api/... | 后端 API（经 nginx 反代） |
| http://localhost:8000/docs | FastAPI Swagger（直连 backend 调试，不经 nginx） |

### 灌数据
```bash
# 热度前 20（本机出网需代理时加 -e HTTPS_PROXY=http://<宿主IP>:端口）
docker compose exec -e TMDB_COUNT=20 -e TMDB_MODE=popular -e TMDB_ONLY_NEW=true backend python -u -m app.crawler.tmdb
# 评分前 20
docker compose exec -e TMDB_COUNT=20 -e TMDB_MODE=top_rated -e TMDB_ONLY_NEW=true backend python -u -m app.crawler.tmdb
# 建语义向量（走国内星火，不需代理）
docker compose exec backend python -u build_embeddings.py
```
> 爬虫是「每部立即落库」，中途中断最多丢当前 1 部；`TMDB_ONLY_NEW=true` 自动跳过已入库影片，避免重复。
> 本机连 TMDb 被墙需代理时，代理地址用 Windows 在 WSL 网络里的真实 IP（非 127.0.0.1），详见 `docker流程.md`。

### 停止 / 重启 / 卸载
```bash
docker compose stop        # 暂停三容器，数据库数据卷保留
docker compose start       # 重新启动
docker compose down        # 删容器（保留数据库数据卷）
docker compose down -v     # ⚠️ 连数据库数据卷一起删，电影数据全没
docker compose up -d --build   # 改了代码/镜像后重建再起（铁律：改代码必带 --build）
```

### 环境变量（根 `.env`，重要，勿提交仓库）
| 变量 | 说明 | 默认 |
|------|------|------|
| `MYSQL_ROOT_PASSWORD` | MySQL root 密码 | `password` |
| `MYSQL_DATABASE` | 库名 | `llm_pro` |
| `DATABASE_URL` | 后端连库地址（compose 内默认覆盖为 `mysql` 服务名） | — |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | 对话大脑（如智谱 GLM-4-Flash） | 留空则 Agent 返回「未启用」 |
| `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` | 星火 Qwen3-Embedding-8B | 见 `.env.example` |
| `TMDB_API_KEY` | TMDb 爬虫 key（灌数据才需要） | 留空 |
| `VITE_BACKEND_URL` | 前端构建期注入的接口基址 | `http://localhost` |

---

## Agent 架构（混合：规则路由为主 + Function Calling 兜底）
> 本系统是**「规则路由 + 工具增强 + LLM 润色」混合体**
> - **主路径（默认）**：确定性意图分类 `_classify_intent` → 规则抽取参数 → 调对应工具（find_movies / get_movie_info_by_name 等）→ 用 LLM 仅做**润色**。可靠、零幻觉、弱模型不背锅。
> - **兜底路径（自主模式，可开关）**：`runtime_flags` 控制的「自主 Agent 模式」下，才用 `llm.bind_tools` 真·Function Calling 让模型自主选工具，并配护栏（未调工具则强制重试，失败回退确定性路由）。平时几乎不触发。
> 这种「orchestrated workflow + targeted tool use」对齐 2025–2026 业界生产主流（Anthropic《Building Effective Agents》），弱模型下不盲信其自路由是正确的工程判断。

- **闲聊 / 感谢 / 身份 / 暂缓**：确定性代码分支（固定自介、不查库等），不交给弱模型发挥。
- **电影相关（主路径）**：规则分类 + 参数抽取 + 工具调用 + LLM 润色；三层保障：
  1. **护栏**：开启自主模式时，涉及电影却没调工具 → 强制重试；重试仍失败 → 退回确定性路由兜底（仍零幻觉）。
  2. **确定性安全网**：从用户原话自动抽取类型词 / 识别已知片名，注入提示，治弱模型「丢类型 / 把片名当助手名」。分类优先级已修正为 **身份 → 片名 → 推荐 → 问候兜底**。
  3. **抗幻觉**：答案 100% 来自工具返回的 DB 数据；库内无此片时**先查库确认、再如实说明**（点名电影 → 基于公开知识简介 + **代码强制追加免责声明**「资料库暂未收录，仅供参考」，不谎称来自资料库；按条件筛选落空 → 直接说明库内暂无）。润色后做**数量声明清洗**（`_sanitize_polish`），强制只报工具真实总数。

## 流式输出（SSE）
- **协议**：后端 `/api/agent/chat/stream` 用 `StreamingResponse(media_type="text/event-stream")` 推送 SSE，带 `X-Accel-Buffering: no` + `Cache-Control: no-cache`；nginx 侧 `proxy_buffering off` 保证逐 token 不被攒批。事件形如 `event: token\ndata: <文本>\n\n`，结束推 `event: done\ndata: <meta JSON>`。
- **两层修复（缺一不可）**：① LLM 层 `ChatOpenAI` 必须 `streaming=True`；② 应用层流式路径直接 `yield from _llm_reply_stream` 逐 token 吐（旧版曾先缓冲再按块吐，导致不实时）。实测 `glm-4-flash` / `glm-4-flash-250414` 均真·流式，模型只影响首字延迟 TTFT。
- **前端消费**：`AgentChat.vue` 用 `fetch` 直连 + `resp.body.getReader()` 按 `\n\n` 切分事件，逐片追加 + 自动滚动（axios 会整段缓冲，故刻意用 fetch）。

## 评测（数据分析展示）
```bash
cd backend
python run_eval.py --quiet     # 真连 LLM 跑 eval_cases.json，结果写 eval_result.json
```
- 22 条用例，覆盖 **库内事实 / 库外电影 / 闲聊 / 身份 / 暂缓 / 类型 / 多维 / 模糊片名 / 标题片段 / 多轮记忆 / 语义** 等类别（每类 2 条）。其中「标题片段」测「标题里带 X 字 → 按片名搜索」，「多轮记忆」测续轮继承（"推荐爱情片 → 再推荐一部"必须仍推爱情片且不同）。
- **评分严格化**：库外电影不再「应拒答」而是「公开知识作答 + 代码免责声明」，评分改为「必须有声明 且 不得谎称来自资料库」；具体电影问答退化成推荐列表即判失败；闲聊须真「接住话」。默认 `temperature=0` 跑基线（可复现，分数稳定）。
- **当前基线（eval_result.json）**：样本总数 22（有效评分 20），**总通过率 100%**，库外诚实声明率 100%，幻觉率 0%。

## 项目扩展（待做）
- [ ] 用户登录 / 注册 / 鉴权
- [ ] 用户打分 / 评论 / 收藏功能
- [ ] 否定类型排除（"不要动画的爱情片"）
- [ ] 片名转写漂移修复（"你的名字。"被写成"你的名字"）
- [ ] MCP 工具标准化 / 多智能体编排
- [ ] 扩用例量、MovieLens 全量或更多片源
