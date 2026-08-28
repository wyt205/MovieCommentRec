# 智影 (CineSage) Docker 部署流程

> 本文档记录本项目的容器化部署与运维命令，按执行顺序整理。
> 环境：WSL2 (Ubuntu) + Docker Engine + Docker Compose。

## 目录

1. [更换镜像源](#1-更换镜像源)
2. [构建与启动](#2-构建与启动)
3. [日常启动 / 关闭](#3-日常启动--关闭)
4. [修复海报字段超限（清脏数据）](#4-修复海报字段超限清脏数据)
5. [运行爬虫灌数据](#5-运行爬虫灌数据)
6. [验证数据](#6-验证数据)
7. [构建语义向量](#7-构建语义向量)
8. [各网址说明](#8-各网址说明)

---

## 1. 更换镜像源

编辑 Docker 守护进程配置：

```bash
sudo vim /etc/docker/daemon.json
```

写入以下内容（使用国内镜像加速拉取）：

```json
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io"
  ]
}
```

修改后**务必重启 Docker 服务**使配置生效：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

> ⚠️ 若某镜像源不稳定（403 / 镜像层损坏），可换阿里云等其它源，并清理坏缓存：
> `docker builder prune -af`

---

## 2. 构建与启动

```bash
docker compose config            # 校验 compose 语法（不报错才继续）
docker compose up -d --build     # 首次构建镜像并后台启动三容器
docker compose ps                # 查看状态（应三容器 Up，mysql healthy）
```

若只改了 backend 代码 / 配置，重建单个服务即可：

```bash
docker compose up -d --force-recreate backend
```

---

## 3. 日常启动 / 关闭

```bash
docker compose start            # 启动（在 docker-compose.yml 所在目录）
docker compose stop             # 停止（保留容器与数据）
# 或指定项目名停止（任意路径均可）：
docker compose -p <项目名> stop
# 或直接按容器名停止单个（绕过 compose）：
docker stop <容器名>
```

> 数据在挂载卷中，`stop` / `down` 都不会丢失；只有 `docker compose down -v` 才会连数据卷一起删除。

---

## 4. 修复海报字段超限（清脏数据）

早期 `poster` 列是 64KB 的 `BLOB`，大海报写入会报 `Data too long`。代码已改为 `LONGBLOB`，但历史旧表需手动修正，并清空首次失败的脏数据：

```bash
docker compose exec mysql mysql -uroot -ppassword llm_pro -e \
"ALTER TABLE movies MODIFY COLUMN poster LONGBLOB; SET FOREIGN_KEY_CHECKS=0; DELETE FROM movie_cast; DELETE FROM movies; SET FOREIGN_KEY_CHECKS=1;"
```

> ⚠️ 命令行明文写密码会出现 `Using a password on the command line...` 警告，属正常现象，不影响执行。

---

## 5. 运行爬虫灌数据

爬虫需经**本机 VPN 代理**出网访问 TMDb。代理地址为 Windows 宿主在 WSL 中的 IP，**每次重启 WSL 可能变化**，请用以下命令获取后替换命令中的 IP：

```bash
grep nameserver /etc/resolv.conf    # 取 nameserver 行，如 172.17.32.1
```

> 💡 关键经验：Docker exec 命令**务必用单行**，不要用多行 `&&` 续行——续行在粘贴时容易丢参数，导致代理没进容器、爬虫卡死无输出。

热度最高 20 部：

```bash
docker compose exec -e HTTP_PROXY=http://172.17.32.1:29290 -e HTTPS_PROXY=http://172.17.32.1:29290 -e TMDB_COUNT=20 -e TMDB_MODE=popular -e TMDB_ONLY_NEW=true backend python -u -m app.crawler.tmdb
```

评分最高 20 部：

```bash
docker compose exec -e HTTP_PROXY=http://172.17.32.1:29290 -e HTTPS_PROXY=http://172.17.32.1:29290 -e TMDB_COUNT=20 -e TMDB_MODE=top_rated -e TMDB_ONLY_NEW=true backend python -u -m app.crawler.tmdb
```

> `-u` 强制实时输出；`-e TMDB_ONLY_NEW=true` 跳过已存在的电影，避免重复入库。

---

## 6. 验证数据

```bash
docker compose exec backend python -u -c "from app.db.database import SessionLocal; from app.models import Movie; print('电影总数=', SessionLocal().query(Movie).count())"

docker compose exec mysql mysql -uroot -ppassword llm_pro -e "SELECT COUNT(*) AS 有海报 FROM movies WHERE poster IS NOT NULL;"
```

---

## 7. 构建语义向量

走国内星火 Embedding API，**无需代理**：

```bash
docker compose exec backend python -u build_embeddings.py
```

验证向量数应等于电影数：

```bash
docker compose exec backend python -u -c "from app.db.database import SessionLocal; from app.models import Movie, MovieEmbedding; s=SessionLocal(); print('电影=', s.query(Movie).count(), ' 向量=', s.query(MovieEmbedding).count())"
```

---

## 8. 各网址说明

| 网址 | 是什么 | 用途 |
| :--- | :--- | :--- |
| **http://localhost/** | 前端主页（nginx 托管） | 用户和 Agent 聊电影的主界面 |
| **http://localhost/admin/** | 管理端 | 看电影库、跑评测、看指标 |
| **http://localhost/api/...** | 后端 API | 前端通过 nginx 反代到这里，正常访问走它 |
| **http://localhost:8000/docs** | FastAPI Swagger | 接口文档，仅走 `:8000` 直连 backend，不经 nginx（nginx 只反代 `/api` 和 `/static`） |
| **http://localhost:8000/** | backend 直连 | 健康检查 / 调试用 |
| **http://localhost/api/movies/{id}/poster** | 海报 | 从 MySQL BLOB 回传图片，前端 `<img>` 用它 |
