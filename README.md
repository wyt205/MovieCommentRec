# llm-pro 影评网站

一个**前后端分离**的影评聚合网站骨架：后端 FastAPI + MySQL，前端 Vue 3 + Vite。
后续将接入 LLM Agent（对话模型 + 向量检索）实现"推荐电影 / 找相似 / 剧透护栏 / 人味闲聊"等能力。

## 目录结构
```
llm-pro/
├── launcher.py           # 总启动器（Tkinter GUI：管理前后端/爬虫/预留功能）
├── sql/
│   └── init.sql            # MySQL 建库建表（movies / reviewers / reviews / short_comments）
├── backend/                # FastAPI 服务
│   ├── app/
│   │   ├── main.py         # 入口 + CORS + 启动建表
│   │   ├── core/config.py  # 读 .env 的 DATABASE_URL
│   │   ├── db/database.py  # SQLAlchemy engine / Session
│   │   ├── models.py       # ORM 模型
│   │   ├── schemas.py      # Pydantic 响应模型
│   │   ├── crud.py         # 增删查（upsert）
│   │   ├── api/            # 路由 movies / reviews / short_comments
│   │   └── crawler/douban.py  # 豆瓣爬虫示例
│   ├── requirements.txt
│   └── .env.example
└── frontend/               # Vue 3 + Vite
    ├── src/
    │   ├── api/            # axios 客户端 + movies / reviews
    │   ├── components/     # NavBar / MovieCard / ReviewCard
    │   ├── views/          # MovieList / MovieDetail
    │   ├── router/         # 路由
    │   ├── App.vue
    │   └── main.js
    ├── vite.config.js      # /api 代理到后端 8000
    └── index.html
```

## 数据来源（豆瓣）
爬取字段对应 `sql/init.sql`：
- **电影**：片名 / 年份 / 导演 / 编剧 / 主演 / 类型 / 国家 / 语言 / 上映日期 / 片长 / 又名 / IMDb / 评分 / 评价人数 / 星级分布 / 简介 / 海报
- **影评**：影评 id / 标题 / 作者(头像+昵称) / 推荐程度(力荐=5…很差=1) / 摘要 / 正文 / 有用数 / 回应数 / 浏览 / 发布时间 / 详情链接
- **短评**：用户 / 评分 / 日期 / 有用数 / 内容

> 爬虫仅用于学习，请遵守豆瓣 robots.txt 与反爬策略。

## 总启动器（launcher.py）
不想手动敲命令？根目录的 `launcher.py` 是一个**零依赖**的桌面 GUI，统一管理本项目：
- 单独 / 一键启动「后端 FastAPI」与「前端 Vue」
- 状态灯显示运行中（绿）/ 已停止（红）
- 「进入」按钮：前端 → http://localhost:5173；后端 → http://localhost:8000/docs
- 选择性停止（停前端 / 停后端 / 停全部）
- **数据库（MySQL）连通测试**：「测试连接」按钮读取 `backend/.env` 的 `DATABASE_URL` 并尝试连接，状态灯（灰=未测 / 绿=成功 / 红=失败）+ 详细错误；「打开 .env」一键生成/编辑连接串
- 启动豆瓣爬虫 + 实时日志 + 进度条
- 预留「管理端」「重跑切片 / 建立向量库」按钮（后续接入即可启用）
- 智能诊断：后端启动失败 / 前端 `vite` 缺失时，日志区自动给出排查建议

运行（推荐在 llm-pro 环境中）：
```bash
conda activate llm-pro
python launcher.py
```
直接双击也行：启动器会自动探测 `llm-pro` conda 环境的 python 来跑后端。

## 启动步骤
### 1. 数据库
```bash
mysql -u root -p < sql/init.sql
```

### 2. 后端
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env      # 填入你的 MySQL 账号密码
uvicorn app.main:app --reload --port 8000
# 文档：http://localhost:8000/docs
```

### 3. 前端
```bash
cd frontend
npm install
npm run dev              # http://localhost:5173
```

## 填充示例数据
已写入 `backend/seed.py`（读 `.env` 的 `DATABASE_URL`，先建库→重建表→插入示例）：
```bash
cd backend
conda activate llm-pro
python seed.py
```
当前库已含示例：**2 部电影（肖申克的救赎 / 星际穿越）、3 位评论者、3 条影评、4 条短评**，可启动前后端直接看效果。

## 图片类数据能存 MySQL 吗？
**技术上能，但不推荐直接存文件本体。**
- MySQL 提供 `BLOB` 系列类型存二进制：`TINYBLOB`(<256B) / `BLOB`(<64KB) / `MEDIUMBLOB`(<16MB) / `LONGBLOB`(<4GB)。把图片 byte 写进去即可。
- **更常见、更优的做法**：表里的 `poster_url` / `avatar_url` 只存**图片的 URL 或文件路径**（如本项目做法），图片文件放 CDN / 对象存储 / 静态目录。优点：库体积小、备份快、可直接浏览器缓存、不占数据库 IO。
- 什么时候才用 BLOB：图片很小（头像/图标）、且必须和记录强一致、不允许外链失效时。本项目海报/头像均用 URL 字段，符合工程惯例。

## 下一步（待做）
- [ ] 跑通豆瓣爬虫，灌入真实数据
- [ ] 接入对话模型（GLM-4-Flash / 星火 Ultra，function calling）
- [ ] 接入向量模型（Qwen3-Embedding-8B）做相似电影检索
- [ ] 用 LangChain `@tool` / `bind_tools` 把"推荐/相似/剧透护栏"封装成 Agent 工具

## 常见问题排查
**Q：点「一键启动」后端报 `Access denied for user 'root'@'localhost'`？**
A：MySQL 账号/密码不对。启动器里先点「测试连接」按钮诊断。然后点「打开 .env」（首次会从 `.env.example` 复制生成），把
`DATABASE_URL=mysql+pymysql://root:password@localhost:3306/llm_pro?charset=utf8mb4`
中的 `password` 改成你本机 MySQL root 的真实密码，保存后再启动后端。

**Q：前端报 `'vite' 不是内部或外部命令`？**
A：`frontend` 目录还没装依赖。先在该目录执行 `npm install`，再启动前端（或一键启动）。

**Q：后端报 `Unknown database 'llm_pro'`？**
A：数据库还没建。先在本机 MySQL 执行 `mysql -u root -p < sql/init.sql` 创建库与表。
