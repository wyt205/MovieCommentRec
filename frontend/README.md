# frontend（Vue 3 + Vite）

影评网站前端，前后端分离，通过 `/api` 调用后端 FastAPI。

## 快速开始
```bash
# 安装依赖
npm install

# 启动开发服务器（默认 5173，/api 已代理到后端 8000）
npm run dev
```
打开 http://localhost:5173

## 页面
- `/` 电影列表（支持按片名搜索）
- `/movie/:id` 电影详情（左侧影评、右侧短评）

## 说明
- API 基址在 `src/api/client.js` 设为 `/api`，由 `vite.config.js` 代理到后端，
  无需处理跨域；生产构建时用 nginx 等反向代理同样转发 `/api` 即可。
- 组件：`components/NavBar.vue`、`MovieCard.vue`、`ReviewCard.vue`
- 视图：`views/MovieList.vue`、`MovieDetail.vue`
