# 智影 (CineSage) · 电影推荐 Agent + 影评网站 + 可观测管理端

一个**前后端分离**的电影应用：后端 FastAPI + MySQL，前端 Vue 3 + Vite，并接入了 **LangChain 真·Function Calling 的自主 Agent**（推荐 / 检索 / 闲聊 / 身份问答），外加一个独立的**可观测管理端**（埋点日志 / 缓存命中 / 评测看板）。

> 项目定位：既是一个能跑的电影推荐产品原型，也是一份面向 **Agent 开发 / AI 应用开发** 岗位的练手作品。

## 目录结构
```
llm-pro/
├── launcher.py              # 总启动器（Tkinter GUI：前后端/爬虫/向量库/管理端 统一管理）
├── sql/init.sql             # MySQL 建库建表（纯 TMDb 体系，无 douban 字段）
├── backend/                 # FastAPI 服务
│   ├── app/
│   │   ├── main.py          # 入口 + CORS(*)+ 启动建表 + 挂载路由
│   │   ├── core/config.py   # 读 .env
│   │   ├── db/database.py   # SQLAlchemy engine / Session / 结构迁移
│   │   ├── models.py        # ORM（含 AgentTrace 埋点表）
│   │   ├── schemas.py       # Pydantic 响应模型
│   │   ├── crud.py          # 增删查（多类型 AND 过滤 / 模糊匹配）
│   │   ├── api/             # movies / reviews / genres / agent / admin
│   │   ├── ai/              # Agent 核心
│   │   │   ├── agent.py     # guardrailed 自主 agent（bind_tools + 护栏 + 确定性安全网）
│   │   │   ├── tools.py     # 5 个 @tool（find_movies / get_movie_info_by_name / semantic_search_movies ...）
│   │   │   ├── rag.py       # 混合检索（向量 + 关键词）+ 置信度阈值
│   │   │   └── embeddings.py# 星火 Qwen3-Embedding-8B 客户端
│   │   └── crawler/tmdb.py  # TMDb 爬虫（电影 / 演员 / 影评）
│   ├── build_embeddings.py  # 建 / 重建电影向量库
│   ├── seed.py              # 最小示例数据
│   ├── eval_cases.json      # 评测用例（22 条，11 类，含 2 条应拒答样本）
│   ├── run_eval.py          # 评测运行器 → eval_result.json
│   ├── requirements.txt
│   └── .env.example
├── frontend/                # Vue 3 + Vite
│   └── src/{api,components,views,router}  # 含 AgentChat.vue 对话页
└── admin/                   # 独立管理端（纯静态，零构建依赖，**默认不提交 git**）
    ├── index.html
    ├── styles.css
    └── app.js               # 四面板：💬对话 / 📜日志 / 📊缓存统计 / 🧪评测
```

## 技术栈
- **后端**：FastAPI + SQLAlchemy + MySQL + LangChain 1.x（`bind_tools`）
- **LLM**：智谱 GLM-4-Flash-250414（对话大脑，OpenAI 兼容）
- **Embedding**：星火 MaaS Qwen3-Embedding-8B（768 维，OpenAI 兼容 `/v2/embeddings`）
- **前端**：Vue 3 + Vite（C 端）；原生 HTML/CSS/JS（管理端 `admin/`）
- **检索**：MySQL 内嵌向量 + Python 暴力余弦 + 关键词混合检索

## 数据来源（TMDb）
- 电影 / 演员 / 影评来自 **TMDb**（`crawler/tmdb.py`），字段对应 `sql/init.sql`：片名 / 年份 / 导演 / 演员 / 类型 / 国家 / 上映日期 / 评分 / 评分人数 / 简介 / 海报(URL) 等。
- 语义检索库：启动器点「建立 / 重建语义向量库」→ `build_embeddings.py` 切片 + 调星火 embedding 入库 `movie_embeddings`。
- 海报 / 头像走 URL 字段（不存 BLOB）。

## 运行方式
### 方式一：启动器（推荐）
```bash
conda activate llm-pro      # 依赖装在 conda 的 llm-pro 环境
python launcher.py
```
GUI 内可：
- 「🚀 一键启动前后端」：纯启动（**不再杀残留进程**，避免误伤你正在用的别的服务）；若 8000/5173/5180 已被旧进程占用，先点「停止全部」再启动
- 「AI 能力」框：重跑 TMDb 爬虫 / 建立语义向量库
- 「管理端」按钮：起 `admin/` 静态服务（5180）并打开浏览器

### 方式二：手动
```bash
# 1. 建库
mysql -u root -p < sql/init.sql

# 2. 后端
cd backend
pip install -r requirements.txt
cp .env.example .env        # 填入 MySQL 账号密码 + LLM/EMBEDDING key
uvicorn app.main:app --reload --port 8000
# 文档：http://localhost:8000/docs

# 3. 前端（C 端）
cd frontend
npm install
npm run dev                 # http://localhost:5173

# 4. 管理端（独立，可选）
cd admin
python -m http.server 5180  # http://localhost:5180
```

## Agent 架构（混合：规则路由为主 + Function Calling 兜底）
> **诚实定位（简历口径，务必照此写）**：本系统是**「规则路由 + 工具增强 + LLM 润色」混合体**，不是纯自主 ReAct agent。
> - **主路径（默认）**：确定性意图分类 `_classify_intent` → 规则抽取参数 → 调对应工具（find_movies / get_movie_info_by_name 等）→ 用 LLM 仅做**润色**。可靠、零幻觉、弱模型不背锅。
> - **兜底路径（自主模式，可开关）**：`runtime_flags` 控制的「自主 Agent 模式」下，才用 `llm.bind_tools` 真·Function Calling 让模型自主选工具，并配护栏（未调工具则强制重试，失败回退确定性路由）。平时几乎不触发。
> 这种「orchestrated workflow + targeted tool use」对齐 2025–2026 业界生产主流（Anthropic《Building Effective Agents》），弱模型下不盲信其自路由是正确的工程判断。

- **闲聊 / 感谢 / 身份 / 暂缓**：确定性代码分支（固定自介、不查库等），不交给弱模型发挥。
- **电影相关（主路径）**：规则分类 + 参数抽取 + 工具调用 + LLM 润色；三层保障：
  1. **护栏**：开启自主模式时，涉及电影却没调工具 → 强制重试；重试仍失败 → 退回确定性路由兜底（仍零幻觉）。
  2. **确定性安全网**：从用户原话自动抽取类型词 / 识别已知片名，注入提示，治弱模型"丢类型 / 把《你的名字》当问助手名"。分类优先级已修正为 **身份 → 片名 → 推荐 → 问候兜底**，避免"你好啊，推荐一部爱情片"被误判成闲聊而凭记忆编造库外电影。
  3. **抗幻觉**：答案 100% 来自工具返回的 DB 数据；库外电影如实拒答，拒答后禁止补外部知识。润色后做**数量声明清洗**（`_sanitize_polish`），强制只报工具真实总数，删掉弱模型自编的"共 N 部"等幻觉。

## 流式输出（SSE · 即 JD 里的"大模型流式开发"）
- **协议**：后端 `/api/agent/chat/stream` 用 `StreamingResponse(media_type="text/event-stream")` 推送
  SSE（Server-Sent Events），并带 `X-Accel-Buffering: no`（防反向代理缓冲）+ `Cache-Control: no-cache`。
  每条事件形如 `event: token\ndata: <文本片段>\n\n`，结束推 `event: done\ndata: <meta JSON>`。
- **关键修复（两层，缺一不可；曾两次误诊）**：
  1. **LLM 层**：LangChain `ChatOpenAI` **必须显式 `streaming=True`**，否则 `.stream()` 不会发起真正的流式 HTTP 请求，而是先完整生成、再整段作为一个 chunk 一次性 yield（表现："思考很久、然后一下子全吐出"）。
  2. **应用层（最容易漏、本次真因）**：流式路径 `_respond_result_stream` 曾为做"数量清洗"把 LLM 整段输出先 `"".join` 缓冲、清洗完再按 2 字/块吐——导致**电影/推荐这条最常用路径完全不实时流式**（长空白 → 再 2 字蹦），用户感知为"流式失效"。现已改为 `yield from _llm_reply_stream` 直接逐 token 吐；数量幻觉改"源头预防"（模板禁止自编数量 + 工具结果已含真实总数），非流式 `_respond_result` 仍保留 `_sanitize_polish` 硬兜底。
  - 两层都修好后实测：任何支持 SSE 的模型（`glm-4-flash` / `glm-4-flash-250414`）都真·逐 token 流式。**模型只影响首字延迟 TTFT、不影响是否流式**——`glm-4-flash-250414` 实测 TTFT≈8.5s、`glm-4-flash`≈2.2s，二者后续都密集流式；要首字快就换 `glm-4-flash`。
- **前端消费**：`AgentChat.vue` 用 `fetch` 直连 `127.0.0.1:8000`（不走 vite 代理）+ `resp.body.getReader()`
  按 `\n\n` 切分事件、`event/data` 解析、逐片 `bot.text +=` 追加 + 自动滚动，实现打字机效果。
  （注意：axios 原生不支持流式、会整段缓冲；本项目刻意用 fetch + getReader 规避该坑。）
- 工具调用循环本身不产出可见文本（工具结果只在后端执行），其间前端显示「正在思考…」。

## 评测（抗幻觉从"嘴上说"变"数据证明"）
```bash
cd backend
python run_eval.py --quiet     # 真连 GLM 跑 eval_cases.json，结果写 eval_result.json
```
- 用例含「库内事实 / 应拒答（库外电影）/ 闲聊 / 身份 / 暂缓 / 类型抽取 / 多维度 / 否定类型 / 模糊片名 / 标题片段 / 语义」11 类，每类 2 条（共 22 条）。其中「标题片段」测更聪明的路由（"标题里带X字"→按片名搜索），「语义」含未知类型兜底（如"治愈系"→语义检索）。
- 指标：编造虚假事实幻觉率、拒答准确率、各维度通过率。
- **已知缺陷（评测已定位，部分已修）**：① 数量幻觉——润色自编"共 N 部"已用 `_sanitize_polish` 清洗（只报工具真实总数）；但**拒答库外电影后仍可能补外部知识（越界泄漏）**尚待修；② "不要动画"仍混入动画片（否定类型不牢）；③ 润色时改写标准片名（片名转写漂移）。
- **评测严谨化（已落地）**：`run_eval.py` 默认固定 `temperature=0` 跑基线——关闭随机性 → 输出确定性、可复现，分数稳定、两次跑可直接对比（免费 GLM-4-Flash 非确定性，不固定温度分数会抖动）。可用 `python run_eval.py --temperature 0.3` 临时覆盖。早期版本的对话响应缓存（`_RESP_CACHE`）已彻底移除，评测不再依赖缓存、每次都真实走工具查询。

## 下一步（待做 · 见上方进度表 🔴🔴）
- [ ] Docker 部署
- [ ] MCP 工具标准化
- [ ] 多智能体编排
- [x] 评测严谨化·temperature=0 确定性基线（已落地；扩大用例 + 修 3 已知缺陷仍待做）

## 常见问题排查
**Q：点「一键启动」后端报 `Access denied for user 'root'@'localhost'`？**
A：MySQL 账号/密码不对。启动器先点「测试连接」诊断，再点「打开 .env」把
`DATABASE_URL=mysql+pymysql://root:password@localhost:3306/llm_pro?charset=utf8mb4`
中的 `password` 改成真实密码后保存重启。

**Q：前端报 `'vite' 不是内部或外部命令`？**
A：`frontend` 还没装依赖，先 `npm install` 再启动。

**Q：后端报 `Unknown database 'llm_pro'`？**
A：库还没建，先 `mysql -u root -p < sql/init.sql`。

**Q：对话像"旧代码 / 回声 / 幻觉"？**
A：几乎必是 **8000 端口有残留旧进程**在顶服务。关掉启动器 → 重开 → 一键启动（会先清场）；或手动杀 8000 残留再起。

## 当前进度（对照通用 Agent 岗位 JD ≈ 72%）

| 状态             | 能力                      | 说明                                                         |
| ---------------- | ------------------------- | ------------------------------------------------------------ |
| ✅ 已完成（核心） | LLM 调用 / Prompt 工程    | 智谱 GLM-4-Flash，system prompt 护栏设计                     |
| ✅ 已完成（核心） | LangChain / Agent 框架    | `bind_tools` 真·Function Calling，非手写 ReAct               |
| ✅ 已完成（核心） | Function Calling 工具调用 | 工具调用链路（规则路由主路径 + `bind_tools` 自主模式兜底，可切换） |
| ✅ 已完成（核心） | RAG 语义检索              | TMDb 数据切片 + 星火 Qwen3-Embedding-8B 向量 + 混合检索 + 置信度阈值 |
| ✅ 已完成（核心） | 抗幻觉护栏                | 答案锚定 DB / 未知如实拒答 / 拒答后禁止补外部知识            |
| ✅ 已完成（核心） | 后端 FastAPI + 前端 Vue 3 | 完整影评网站（详情 / 评分 / 评论）                           |
| ✅ 已完成（核心） | 数据库建模                | MySQL（movies / reviewers / reviews / movie_cast / movie_embeddings / agent_traces） |
| ✅ 已完成（核心） | 可观测管理端              | 独立 `admin/` 静态站点，四面板（对话 / 日志 / 缓存统计 / 评测） |
| 🟡 部分完成       | 可观测 / 评测             | 评测已固定 temperature=0 基线（确定性、可复现）；3 个已知缺陷与扩大用例仍待做 |
| ✅ 已完成（核心） | 流式输出（大模型流式开发）| SSE 逐 token 推送 + 前端打字机；工具循环期间显示"正在思考…" |
| ✅ 已完成（核心） | 对话记录列表              | 左侧会话列表持久化（DB-backed），可新建 / 选择继续聊 / 删除；切换页面不丢 |
| ✅ 已完成（核心） | 长期记忆（用户偏好）      | 规则抽取喜好落库 user_preferences，注入 system prompt 主动参考（如"喜欢科幻"） |
| 🔴 待完成         | 部署 / 容器化             | Docker 化（当前仅本地启动器）                                |
| 🔴 待完成         | MCP 工具标准化            | 把工具封装成 MCP server                                      |
| 🔴 待完成         | 多智能体编排              | 单 agent，未做多 agent 协作                                  |
| 🟡 部分完成       | 评测严谨化·剩余项         | temperature=0 基线已落地 ✅；扩大用例 + 修 3 已知缺陷仍待做（对话缓存已移除，无需"缓存生效"） |

> 📌 **下次接力重点（用户计划完成 🔴 红 + 🟡 灰 部分）**：Docker 部署、MCP、多智能体、评测严谨化（流式输出 / 对话记录 / 长期记忆已完成 ✅）。
> 简历表述提醒：**不要写"零幻觉"**，写"可溯源 + 未知如实拒答，实测编造虚假事实 0%"；核心能力讲成硬货，空白项讲"规划中 / 可快速补齐"。