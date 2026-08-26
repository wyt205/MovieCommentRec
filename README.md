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
│   ├── eval_cases.json      # 评测用例（29 条，含应拒答样本）
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
- **LLM**：智谱 GLM-4-Flash（对话大脑，OpenAI 兼容）
- **Embedding**：星火 MaaS Qwen3-Embedding-8B（768 维，OpenAI 兼容 `/v2/embeddings`）
- **前端**：Vue 3 + Vite（C 端）；原生 HTML/CSS/JS（管理端 `admin/`）
- **检索**：MySQL 内嵌向量 + Python 暴力余弦 + 关键词混合检索

## 数据来源（TMDb，非豆瓣）
- 电影 / 演员 / 影评来自 **TMDb**（`crawler/tmdb.py`），字段对应 `sql/init.sql`：片名 / 年份 / 导演 / 演员 / 类型 / 国家 / 上映日期 / 评分 / 评分人数 / 简介 / 海报(URL) 等。
- 语义检索库：启动器点「建立 / 重建语义向量库」→ `build_embeddings.py` 切片 + 调星火 embedding 入库 `movie_embeddings`（当前约 40 部）。
- 海报 / 头像走 URL 字段（不存 BLOB），符合工程惯例。

## 运行方式
### 方式一：启动器（推荐）
```bash
conda activate llm-pro      # 依赖装在 conda 的 llm-pro 环境
python launcher.py
```
GUI 内可：
- 「🚀 一键启动前后端」：先清干净 8000/5173/5180 残留进程再起，避免打到旧代码
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

## Agent 架构（guardrailed 自主 agent）
- **闲聊 / 感谢 / 身份 / 暂缓**：确定性代码分支（固定自介、不查库等），不交给弱模型发挥。
- **电影相关**：模型通过 `bind_tools` **自主决定**调哪个工具、可多步串联；代码三层保障：
  1. **护栏**：涉及电影却没调工具 → 强制重试；重试仍失败 → 退回确定性路由兜底（仍零幻觉）。
  2. **确定性安全网**：从用户原话自动抽取类型词 / 识别已知片名，注入提示，治弱模型"丢类型 / 把《你的名字》当问助手名"。
  3. **抗幻觉**：答案 100% 来自工具返回的 DB 数据；库外电影如实拒答，拒答后禁止补外部知识。

## 评测（抗幻觉从"嘴上说"变"数据证明"）
```bash
cd backend
python run_eval.py --quiet     # 真连 GLM 跑 eval_cases.json，结果写 eval_result.json
```
- 用例含「库内事实 / 应拒答（库外电影）/ 闲聊 / 身份 / 暂缓 / 类型抽取 / 多维度 / 否定类型 / 模糊 / 语义」9 类。
- 指标：编造虚假事实幻觉率、拒答准确率、各维度通过率。
- **已知缺陷（评测已定位，待修）**：① 拒答库外电影后仍补外部知识（越界泄漏）；② "不要动画"仍混入动画片（否定类型不牢）；③ 润色时改写标准片名（片名转写漂移）。
- 注：免费 GLM-4-Flash 非确定性，单次分数会抖动；严谨评测应固定 `temperature=0` 跑基线。

## 下一步（待做 · 见上方进度表 🔴🔴）
- [ ] 流式输出（SSE 打字机）
- [ ] 长期记忆（持久化用户偏好，下次主动推荐）
- [ ] Docker 部署
- [ ] MCP 工具标准化
- [ ] 多智能体编排
- [ ] 评测严谨化（temperature=0 基线 + 扩大用例 + 修上述 3 个已知缺陷）

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

## 当前进度（对照通用 Agent 岗位 JD ≈ 60%）

| 状态             | 能力                      | 说明                                                         |
| ---------------- | ------------------------- | ------------------------------------------------------------ |
| ✅ 已完成（核心） | LLM 调用 / Prompt 工程    | 智谱 GLM-4-Flash，system prompt 护栏设计                     |
| ✅ 已完成（核心） | LangChain / Agent 框架    | `bind_tools` 真·Function Calling，非手写 ReAct               |
| ✅ 已完成（核心） | Function Calling 工具调用 | 模型自主选工具 + 多步串联 + 护栏兜底                         |
| ✅ 已完成（核心） | RAG 语义检索              | TMDb 数据切片 + 星火 Qwen3-Embedding-8B 向量 + 混合检索 + 置信度阈值 |
| ✅ 已完成（核心） | 抗幻觉护栏                | 答案锚定 DB / 未知如实拒答 / 拒答后禁止补外部知识            |
| ✅ 已完成（核心） | 后端 FastAPI + 前端 Vue 3 | 完整影评网站（详情 / 评分 / 评论）                           |
| ✅ 已完成（核心） | 数据库建模                | MySQL（movies / reviewers / reviews / movie_cast / movie_embeddings / agent_traces） |
| ✅ 已完成（核心） | 可观测管理端              | 独立 `admin/` 静态站点，四面板（对话 / 日志 / 缓存统计 / 评测） |
| 🟡 部分完成       | 可观测 / 评测             | 已落地，但评测尚非严谨（弱模型非确定性，单次分数会抖动，需固定 temperature 基线） |
| 🔴 待完成         | 流式输出                  | SSE 打字机效果                                               |
| 🔴 待完成         | 长期记忆                  | 持久化用户偏好（当前仅会话级内存）                           |
| 🔴 待完成         | 部署 / 容器化             | Docker 化（当前仅本地启动器）                                |
| 🔴 待完成         | MCP 工具标准化            | 把工具封装成 MCP server                                      |
| 🔴 待完成         | 多智能体编排              | 单 agent，未做多 agent 协作                                  |
| 🔴 待完成         | 评测严谨化                | 固定 temperature 基线 + 扩大用例 + 缓存真正生效              |

> 📌 **下次接力重点（用户计划完成 🔴 红 + 🟡 灰 部分）**：流式输出、长期记忆、Docker 部署、MCP、多智能体、评测严谨化。
> 简历表述提醒：**不要写"零幻觉"**，写"可溯源 + 未知如实拒答，实测编造虚假事实 0%"；核心能力讲成硬货，空白项讲"规划中 / 可快速补齐"。