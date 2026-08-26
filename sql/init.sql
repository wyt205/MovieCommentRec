-- ============================================================
-- llm-pro 影评网站 - MySQL 初始化脚本
-- 数据库：llm_pro
-- 字符集：utf8mb4（支持中文 + emoji）
-- 纯 TMDb 结构：电影 / 类型目录 / 评论者 / 影评（无豆瓣字段、无短评表）
-- ============================================================

CREATE DATABASE IF NOT EXISTS `llm_pro`
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE `llm_pro`;

-- 先关闭外键检查，避免 DROP 顺序受外键依赖限制（child 表引用 parent 表时）
SET FOREIGN_KEY_CHECKS = 0;

-- ------------------------------------------------------------
-- 1. 电影表 movies
--    对应 TMDb 电影详情（api.themoviedb.org）
-- ------------------------------------------------------------
DROP TABLE IF EXISTS `movie_cast`;
DROP TABLE IF EXISTS `reviews`;
DROP TABLE IF EXISTS `reviewers`;
DROP TABLE IF EXISTS `movies`;
DROP TABLE IF EXISTS `genres`;

SET FOREIGN_KEY_CHECKS = 1;

CREATE TABLE `movies` (
  `id`              BIGINT       NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tmdb_id`         VARCHAR(32)  DEFAULT NULL COMMENT 'TMDb movie id（唯一来源）',
  `title`           VARCHAR(255) NOT NULL COMMENT '片名',
  `original_title`  VARCHAR(255) DEFAULT NULL COMMENT '原名',
  `year`            INT          DEFAULT NULL COMMENT '年份',
  `directors`       VARCHAR(512) DEFAULT NULL COMMENT '导演（多名用逗号分隔）',
  `writers`         VARCHAR(512) DEFAULT NULL COMMENT '编剧',
  `casts`           VARCHAR(1024) DEFAULT NULL COMMENT '主演',
  `genres`          VARCHAR(255) DEFAULT NULL COMMENT '类型（逗号分隔，与 genres 表一致）',
  `country`         VARCHAR(255) DEFAULT NULL COMMENT '制片国家/地区',
  `language`        VARCHAR(255) DEFAULT NULL COMMENT '语言',
  `release_date`    VARCHAR(255) DEFAULT NULL COMMENT '上映日期（可能多段）',
  `runtime`         VARCHAR(128) DEFAULT NULL COMMENT '片长',
  `aka`             VARCHAR(512) DEFAULT NULL COMMENT '又名',
  `imdb`            VARCHAR(32)  DEFAULT NULL COMMENT 'IMDb 编号',
  `rating`          DECIMAL(3,1) DEFAULT NULL COMMENT '评分（TMDb vote_average，0-10）',
  `rating_count`    INT          DEFAULT NULL COMMENT '评分人数（vote_count）',
  `popularity`      FLOAT        DEFAULT NULL COMMENT '热度值（TMDb popularity，用于热度排行/排序）',
  `summary`         TEXT         DEFAULT NULL COMMENT '剧情简介（overview）',
  `poster_url`      VARCHAR(512) DEFAULT NULL COMMENT '海报访问地址（指向 /api/movies/{id}/poster）',
  `poster`          LONGBLOB     DEFAULT NULL COMMENT '海报图片字节（存库，不落盘到文件夹）',
  `source`          VARCHAR(16)  DEFAULT NULL COMMENT '数据来源（当前固定 tmdb）',
  `tagline`         VARCHAR(512) DEFAULT NULL COMMENT '宣传语（TMDb tagline）',
  `created_at`      DATETIME     DEFAULT CURRENT_TIMESTAMP COMMENT '入库时间',
  `updated_at`      DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_tmdb_id` (`tmdb_id`),
  KEY `idx_title` (`title`),
  KEY `idx_year` (`year`),
  KEY `idx_popularity` (`popularity`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='电影基本信息（TMDb）';

-- ------------------------------------------------------------
-- 1.5 电影类型目录 genres（TMDb 官方 19 类，单一数据源）
--     爬虫按官方 id 映射中文名存库；前端分类标签从 /api/genres 拉取
-- ------------------------------------------------------------
CREATE TABLE `genres` (
  `id`      INT         NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tmdb_id` INT         NOT NULL COMMENT 'TMDb 官方类型 id',
  `name`    VARCHAR(32) NOT NULL COMMENT '类型中文名',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_tmdb_genre` (`tmdb_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='电影类型目录（TMDb 官方 19 类）';

-- ------------------------------------------------------------
-- 2. 评论者表 reviewers（影评作者，按昵称去重）
-- ------------------------------------------------------------
CREATE TABLE `reviewers` (
  `id`          BIGINT       NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `name`        VARCHAR(128) NOT NULL COMMENT '昵称',
  `avatar_url`  VARCHAR(512) DEFAULT NULL COMMENT '头像链接',
  `location`    VARCHAR(128) DEFAULT NULL COMMENT '常居地',
  `signature`   VARCHAR(512) DEFAULT NULL COMMENT '个性签名',
  `created_at`  DATETIME     DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='评论者';

-- ------------------------------------------------------------
-- 3. 影评表 reviews（TMDb reviews；短评是豆瓣独有概念，本项目不含）
--    一条影评属于一部电影、一个作者
-- ------------------------------------------------------------
CREATE TABLE `reviews` (
  `id`              BIGINT       NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `tmdb_review_id`  VARCHAR(64)  DEFAULT NULL COMMENT 'TMDb review id（唯一来源）',
  `movie_id`        BIGINT       NOT NULL COMMENT '关联电影',
  `reviewer_id`     BIGINT       DEFAULT NULL COMMENT '关联评论者',
  `source`          VARCHAR(16)  DEFAULT 'tmdb' COMMENT '评论来源：tmdb（爬取）/ user（用户发布）',
  `title`           VARCHAR(512) NOT NULL COMMENT '影评标题',
  `rating`          TINYINT      DEFAULT NULL COMMENT '评分(1-5)，由 TMDb 1-10 折算',
  `rating_label`    VARCHAR(8)   DEFAULT NULL COMMENT '推荐程度文字：力荐/推荐/还行/较差/很差',
  `summary`         TEXT         DEFAULT NULL COMMENT '影评摘要（列表页短内容）',
  `content`         LONGTEXT     DEFAULT NULL COMMENT '影评正文（详情页全文）',
  `useful_count`    INT          DEFAULT 0 COMMENT '有用数',
  `comments_count`  INT          DEFAULT 0 COMMENT '回应/评论数',
  `views`           INT          DEFAULT 0 COMMENT '浏览数',
  `publish_date`    DATETIME     DEFAULT NULL COMMENT '发布时间',
  `created_at`      DATETIME     DEFAULT CURRENT_TIMESTAMP,
  `updated_at`      DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_tmdb_review_id` (`tmdb_review_id`),
  KEY `idx_movie` (`movie_id`),
  KEY `idx_reviewer` (`reviewer_id`),
  KEY `idx_publish` (`publish_date`),
  CONSTRAINT `fk_review_movie` FOREIGN KEY (`movie_id`) REFERENCES `movies` (`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_review_reviewer` FOREIGN KEY (`reviewer_id`) REFERENCES `reviewers` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='影评（TMDb / 用户）';

-- ------------------------------------------------------------
-- 3.5 电影演员表 movie_cast（结构化：演员 + 饰演角色 + 头像路径）
--     由爬虫从 TMDb /movie/{id}/credits 的 cast 写入；详情页渲染「演员表」
-- ------------------------------------------------------------
CREATE TABLE `movie_cast` (
  `id`              BIGINT       NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `movie_id`        BIGINT       NOT NULL COMMENT '关联电影',
  `tmdb_person_id`  VARCHAR(32)  DEFAULT NULL COMMENT 'TMDb 人物 id',
  `name`            VARCHAR(128) NOT NULL COMMENT '演员名',
  `character`       VARCHAR(128) DEFAULT NULL COMMENT '饰演角色',
  `order`           INT          DEFAULT 0 COMMENT '排序（主演在前）',
  `profile_path`    VARCHAR(255) DEFAULT NULL COMMENT 'TMDb 头像路径（按需代理加载）',
  `created_at`      DATETIME     DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_movie_cast` (`movie_id`),
  CONSTRAINT `fk_cast_movie` FOREIGN KEY (`movie_id`) REFERENCES `movies` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='电影演员表';

-- ------------------------------------------------------------
-- 评分映射参考（爬虫把 TMDb 1-10 折算成 1-5 推荐度）：
--   力荐 = 5, 推荐 = 4, 还行 = 3, 较差 = 2, 很差 = 1
-- ------------------------------------------------------------
