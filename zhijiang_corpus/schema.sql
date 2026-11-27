-- ============================================================================
-- 枝江直播回放 AI 字幕语料库 - SQLite Schema (v1)
-- 设计原则：
--   1. 本地 SQLite 开箱即用，支持几百 MB ~ 数 GB 级语料与毫秒级中文检索；
--   2. 字段命名清晰、规范化，便于后续平滑迁移至 PostgreSQL / ClickHouse / Meilisearch；
--   3. videos（场次/分P元数据 + 抓取状态 + 场景打标审计）与 subtitles（核心字幕语料）双表协同，
--      并保证 subtitles 直接携带全部核心检索字段，查询与导出无需复杂联表。
-- ============================================================================

-- 1. 视频分 P 元数据与抓取/场景打标状态表
CREATE TABLE IF NOT EXISTS videos (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    bvid              TEXT    NOT NULL,
    page              INTEGER NOT NULL DEFAULT 1,
    cid               INTEGER,
    up_uid            INTEGER,
    up_name           TEXT    NOT NULL,
    raw_up_name       TEXT    NOT NULL DEFAULT '',
    title             TEXT    NOT NULL DEFAULT '',
    part_title        TEXT    NOT NULL DEFAULT '',
    pub_date          TEXT    NOT NULL,                  -- ISO 日期 YYYY-MM-DD
    pub_ts            INTEGER NOT NULL DEFAULT 0,        -- 原始发布时间戳（秒）
    duration_sec      INTEGER NOT NULL DEFAULT 0,        -- 分 P 时长（秒）
    subtitle_lan      TEXT    NOT NULL DEFAULT '',       -- 抓取到的字幕轨，如 ai-zh / zh-CN / local-srt
    subtitle_status   TEXT    NOT NULL DEFAULT 'pending',-- pending | ok | no_subtitle | failed
    line_count        INTEGER NOT NULL DEFAULT 0,        -- 已入库字幕条数
    scene_type        TEXT    NOT NULL DEFAULT 'unknown',-- unknown | solo | duo | group | collab | special
    scene_confidence  REAL    NOT NULL DEFAULT 0.0,      -- 初筛置信度 0.0 ~ 1.0
    scene_reason      TEXT    NOT NULL DEFAULT '',       -- 规则初筛依据（JSON 摘要）
    scene_locked      INTEGER NOT NULL DEFAULT 0,        -- 1=人工确认锁定，自动初筛不覆盖
    voiceprint_status TEXT    NOT NULL DEFAULT '',       -- '' | ok | skipped | failed（声纹发言人标注状态）
    is_visible        INTEGER NOT NULL DEFAULT 1,        -- 1=可见，0=隐藏/软删除
    extra_meta        TEXT    NOT NULL DEFAULT '{}',     -- 扩展字段（JSON：简介摘要、识别到的出场成员等）
    created_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE (bvid, page)
);

CREATE INDEX IF NOT EXISTS idx_videos_up_date
    ON videos (up_name, pub_date DESC);

CREATE INDEX IF NOT EXISTS idx_videos_status
    ON videos (subtitle_status);

CREATE INDEX IF NOT EXISTS idx_videos_scene_vis
    ON videos (scene_type, is_visible);


-- 2. 核心 AI 字幕语料表
-- 包含全部必需字段：up_name, bvid, page, pub_date, start_sec, end_sec, content, scene_type, is_visible
CREATE TABLE IF NOT EXISTS subtitles (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    up_name           TEXT    NOT NULL,
    bvid              TEXT    NOT NULL,
    page              INTEGER NOT NULL DEFAULT 1,
    pub_date          TEXT    NOT NULL,                  -- YYYY-MM-DD
    start_sec         REAL    NOT NULL,                  -- 起始秒数（保留 2 位小数）
    end_sec           REAL    NOT NULL,                  -- 结束秒数（保留 2 位小数）
    content           TEXT    NOT NULL,                  -- 清洗后的可读字幕正文
    content_fts       TEXT    NOT NULL DEFAULT '',       -- 中英混排字符级分词文本（供 FTS5 索引）
    scene_type        TEXT    NOT NULL DEFAULT 'unknown',
    speaker_name      TEXT    NOT NULL DEFAULT '',       -- 声纹识别发言人（如 嘉然 / 嘉然+乃琳 / 其他/原声 / 空=未标注）
    speaker_conf      REAL    NOT NULL DEFAULT 0.0,      -- 声纹识别置信度 0.0 ~ 1.0
    is_visible        INTEGER NOT NULL DEFAULT 1,
    source            TEXT    NOT NULL DEFAULT 'bilibili_ai', -- bilibili_ai | bilibili_cc | local_import | whisper
    created_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    -- 去重约束：同一视频同一分 P 下相同起止时间与文本不重复插入
    UNIQUE (bvid, page, start_sec, end_sec, content)
);

-- 用于按视频时间轴读取上下文、切片定位、增量覆盖
CREATE INDEX IF NOT EXISTS idx_subtitles_bvid_page_start
    ON subtitles (bvid, page, start_sec);

-- 用于按可见性 + UP 主 + 日期 + 场景类型过滤
CREATE INDEX IF NOT EXISTS idx_subtitles_filter
    ON subtitles (is_visible, up_name, pub_date, scene_type);

CREATE INDEX IF NOT EXISTS idx_subtitles_pub_date
    ON subtitles (pub_date);

CREATE INDEX IF NOT EXISTS idx_subtitles_scene_vis
    ON subtitles (scene_type, is_visible);


-- 3. SQLite FTS5 全文检索索引（外部内容表，零正文冗余 + 触发器自动同步）
-- 配合 tokenizer.py 的 CJK 字符级切分，可对 1字 / 2字 / 长短语 / 英文词均实现毫秒级精确子串匹配
CREATE VIRTUAL TABLE IF NOT EXISTS subtitles_fts USING fts5(
    content_fts,
    content='subtitles',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 0'
);

CREATE TRIGGER IF NOT EXISTS trg_subtitles_ai AFTER INSERT ON subtitles BEGIN
    INSERT INTO subtitles_fts(rowid, content_fts)
    VALUES (new.id, new.content_fts);
END;

CREATE TRIGGER IF NOT EXISTS trg_subtitles_ad AFTER DELETE ON subtitles BEGIN
    INSERT INTO subtitles_fts(subtitles_fts, rowid, content_fts)
    VALUES ('delete', old.id, old.content_fts);
END;

CREATE TRIGGER IF NOT EXISTS trg_subtitles_au AFTER UPDATE OF content_fts ON subtitles BEGIN
    INSERT INTO subtitles_fts(subtitles_fts, rowid, content_fts)
    VALUES ('delete', old.id, old.content_fts);
    INSERT INTO subtitles_fts(rowid, content_fts)
    VALUES (new.id, new.content_fts);
END;


-- 4. 抓取/入库运行日志（便于排查失败场次与断点重试）
CREATE TABLE IF NOT EXISTS ingest_logs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT    NOT NULL,
    bvid              TEXT    NOT NULL,
    page              INTEGER NOT NULL DEFAULT 0,
    status            TEXT    NOT NULL,                  -- ok | skipped | no_subtitle | failed
    detail            TEXT    NOT NULL DEFAULT '',
    created_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_ingest_logs_bvid
    ON ingest_logs (bvid, created_at DESC);
