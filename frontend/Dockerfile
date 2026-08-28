# 智影 前端镜像（多阶段构建：Node 构建 Vue SPA → nginx 托管）
# 构建上下文 = 项目根目录（需在 docker-compose.yml 里指定 context: .）

# ---------- 构建阶段 ----------
FROM node:20-alpine AS build
WORKDIR /app

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./

# 流式对话基址（构建期注入；浏览器访问入口，默认同源 nginx:80）
ARG VITE_BACKEND_URL=http://localhost
ENV VITE_BACKEND_URL=$VITE_BACKEND_URL

RUN npm run build

# ---------- 运行阶段 ----------
FROM nginx:alpine

# Vue SPA 产物
COPY --from=build /app/dist /usr/share/nginx/html
# 管理端纯静态文件
COPY admin /usr/share/nginx/html/admin
# nginx 配置（代理 /api 与 /static 到后端，关闭缓冲以保 SSE 流式）
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
