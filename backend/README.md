# backend（FastAPI）

影评网站后端，提供电影 / 影评 / 短评的 REST API，数据存 MySQL。

## 目录结构
```
backend/
├── app/
│   ├── main.py            # FastAPI 入口 + CORS + 自动建表
│   ├── core/config.py     # 读取 .env 的 DATABASE_URL
│   ├── db/database.py     # SQLAlchemy engine / SessionLocal / Base
│   ├── models.py          # ORM 模型（movies/reviewers/reviews/short_comments）
│   ├── schemas.py         # Pydantic 响应模型
│   ├── crud.py            # 增删查（含 upsert）
│   ├── api/               # 路由
│   │   ├── movies.py
│   │   ├── reviews.py
│   │   └── short_comments.py
│   └── crawler/
│       └── douban.py      # 豆瓣爬虫示例（抓取→解析→入库）
├── requirements.txt
└── .env.example
```

## 快速开始
```bash
# 1. 建库（见 ../sql/init.sql）
mysql -u root -p < ../sql/init.sql

# 2. 安装依赖（建议用虚拟环境）
pip install -r requirements.txt

# 3. 配置连接
cp .env.example .env
# 编辑 .env，填入你的 MySQL 账号密码

# 4. 启动
uvicorn app.main:app --reload --port 8000
```
接口文档：http://localhost:8000/docs

## API 一览（前缀 /api）
- `GET /api/movies?keyword=&skip=&limit=` 电影列表
- `GET /api/movies/{id}` 电影详情（含前 20 条影评）
- `GET /api/reviews?movie_id=&skip=&limit=` 影评列表
- `GET /api/reviews/{id}` 单条影评
- `GET /api/short-comments?movie_id=&skip=&limit=` 短评列表

## 爬虫
```bash
# 先在 douban.py 底部改 subject_id，再运行
python -m app.crawler.douban
```
注意遵守豆瓣反爬与 robots.txt，仅用于学习。
