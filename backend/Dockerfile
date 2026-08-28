# 智影 (CineSage) 后端镜像
# 用法（在 backend/ 目录下）：docker build -t zhiying-backend .
FROM python:3.11-slim

WORKDIR /app

# 少量系统依赖，防止个别包在 slim 镜像里需要本地编译
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制整个 backend（app/ 源码 + 脚本）。.dockerignore 已排除 .env / 缓存 / 陈旧海报
COPY . .

# 海报缓存目录：启动时由 backend 自动从库内 BLOB 预热写入，可重建
RUN mkdir -p static/posters

EXPOSE 8000

# 监听 0.0.0.0 以便容器外（同 compose 网络的 nginx）访问
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
