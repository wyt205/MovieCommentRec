# frontend（C 端：Vue 3 + Vite）

影评网站 + 电影推荐 Agent 的前端，前后端分离，通过 `/api` 调用后端 FastAPI。
生产环境由 `frontend/Dockerfile`（多阶段构建）打包进 nginx 镜像，与后端、MySQL
通过根目录 `docker-compose.yml` 一起编排。

## 本地开发
```bash
npm install
npm run dev          # http://localhost:5173
```
> 开发期 `/api` 由 `vite.config.js` 代理到后端 `8000`，无需处理跨域。

## 页面 / 路由（vue-router，history 模式）
| 路由 | 视图 | 说明 |
|------|------|------|
| `/` | MovieList | 电影列表（按片名/类型搜索、热搜榜 `RankingPanel`） |
| `/movie/:id` | MovieDetail | 电影详情（左侧影评、右侧短评、`StarRating` 评分） |
| `/review/:id` | ReviewDetail | 单条影评详情 |
| `/reviews` | ReviewsIndex | 影评索引 |
| `/agent` | AgentChat | Agent 对话页（SSE 打字机，fetch + getReader 消费流式） |
| `/about` | About | 关于 |

组件：`NavBar.vue`（导航 + 品牌「智影 CineSage」）、`MovieCard.vue`、`ReviewCard.vue`、
`RankingPanel.vue`、`StarRating.vue`、`AppFooter.vue`。

## 与后端交互
- API 客户端：`src/api/`（`client.js` 设基址 `/api`，`movies.js` / `reviews.js` / `agent.js`）。
- 接口基址由构建期注入的 `VITE_BACKEND_URL` 决定（默认 `http://localhost`，即同源 nginx）；
  浏览器侧所有请求走 `/api`，由 nginx 反代到 `backend:8000`（容器网络内）。
- 海报/演员头像经 `/api/movies/{id}/poster`、`/api/movies/cast-photo` 回传（后端从 MySQL BLOB 取）。

## Docker 构建说明
`frontend/Dockerfile`（构建上下文 = 项目根，由 `docker-compose.yml` 指定 `context: .`）：
1. `node:20-alpine` 阶段 `npm ci` + `npm run build` 产出 `dist/`；
2. `nginx:alpine` 阶段把 `dist/` 放 `/usr/share/nginx/html`，把仓库根 `admin/` 放
   `/usr/share/nginx/html/admin`，并用 `frontend/nginx.conf` 接管 80 端口：

- SPA history 路由回退到 `index.html`；
- `/admin/` 静态托管管理端；
- `/api/`、`/static/` 反代到 `backend:8000`，并 `proxy_buffering off` 保证 Agent 流式不被攒批。

构建由 `docker compose up -d --build` 自动触发，无需手动 build；构建期可通过
`VITE_BACKEND_URL` 覆盖接口基址。
