"""SQLite 数据库连接、建库、事务封装与增量写入工具."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .config import VALID_SCENE_TYPES
from .tokenizer import tokenize_for_fts

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


@dataclass
class VideoRecord:
    bvid: str
    page: int
    up_name: str
    pub_date: str
    cid: Optional[int] = None
    up_uid: Optional[int] = None
    raw_up_name: str = ""
    title: str = ""
    part_title: str = ""
    pub_ts: int = 0
    duration_sec: int = 0
    subtitle_lan: str = ""
    subtitle_status: str = "pending"
    line_count: int = 0
    scene_type: str = "unknown"
    scene_confidence: float = 0.0
    scene_reason: str = ""
    scene_locked: int = 0
    voiceprint_status: str = ""
    is_visible: int = 1
    extra_meta: Dict[str, Any] | str = "{}"


@dataclass
class SubtitleRow:
    up_name: str
    bvid: str
    page: int
    pub_date: str
    start_sec: float
    end_sec: float
    content: str
    scene_type: str = "unknown"
    speaker_name: str = ""
    speaker_conf: float = 0.0
    is_visible: int = 1
    source: str = "bilibili_ai"


VALID_SPEAKER_ATOMS = ("嘉然", "贝拉", "乃琳", "向晚", "心宜", "思诺", "珈乐", "其他")


def normalize_speaker_label(raw: str) -> str:
    """校验并规整人工或声纹发言人标签（允许空串清空、单成员、其他、或 A+B 组合）."""
    text = (raw or "").strip()
    if not text:
        return ""
    # 允许用 + / 、 , 空格 分隔多人组合，统一规整为 '+' 连接
    import re

    parts = [p.strip() for p in re.split(r"[+/、,，\s]+", text) if p.strip()]
    if not parts:
        return ""
    dedup: List[str] = []
    for p in parts:
        if p not in VALID_SPEAKER_ATOMS:
            raise ValueError(
                f"无效发言人标签 {p!r}，可选值：{', '.join(VALID_SPEAKER_ATOMS)}（支持 + 组合，如 嘉然+贝拉，留空表示清空）"
            )
        if p not in dedup:
            dedup.append(p)
    if "其他" in dedup and len(dedup) > 1:
        raise ValueError("「其他」不可与成员名同时组合")
    return "+".join(dedup)


class CorpusDB:
    """SQLite 语料库访问层（保持纯标准 SQL，便于后续抽象为 PostgreSQL 适配层）."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=60.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 60000;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        return conn

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self.session() as conn:
            conn.executescript(schema_sql)
            # 无损向后兼容迁移：为历史数据库自动补充声纹标注列（SQLite 元数据级 ALTER，毫秒完成）
            v_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(videos)").fetchall()}
            if "voiceprint_status" not in v_cols:
                conn.execute("ALTER TABLE videos ADD COLUMN voiceprint_status TEXT NOT NULL DEFAULT ''")
            s_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(subtitles)").fetchall()}
            if "speaker_name" not in s_cols:
                conn.execute("ALTER TABLE subtitles ADD COLUMN speaker_name TEXT NOT NULL DEFAULT ''")
            if "speaker_conf" not in s_cols:
                conn.execute("ALTER TABLE subtitles ADD COLUMN speaker_conf REAL NOT NULL DEFAULT 0.0")

    def get_video(self, bvid: str, page: int = 1) -> Optional[Dict[str, Any]]:
        with self.session() as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE bvid = ? AND page = ?",
                (bvid, page),
            ).fetchone()
            return dict(row) if row else None

    def is_video_ingested(self, bvid: str, page: int = 1) -> bool:
        with self.session() as conn:
            row = conn.execute(
                "SELECT subtitle_status FROM videos WHERE bvid = ? AND page = ?",
                (bvid, page),
            ).fetchone()
            return bool(row and row["subtitle_status"] == "ok")

    def is_bvid_completed(self, bvid: str, *, allow_no_subtitle: bool = True) -> bool:
        """判断某个 BV 号是否已完整入库（避免断点续抓时重复请求 /x/web-interface/view）."""
        done_statuses = ("ok", "no_subtitle", "asr_empty") if allow_no_subtitle else ("ok", "asr_empty")
        placeholders = ",".join("?" for _ in done_statuses)
        with self.session() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN subtitle_status IN ({placeholders}) THEN 1 ELSE 0 END) AS done
                  FROM videos
                 WHERE bvid = ?
                """,
                (*done_statuses, bvid),
            ).fetchone()
            total = int(row["total"] or 0)
            done = int(row["done"] or 0)
            return total > 0 and total == done

    def list_retry_bvids(self, include_no_subtitle: bool = False) -> List[str]:
        """查询需要重试的 BV 号列表（pending / failed，可选包含 no_subtitle）."""
        statuses = ["pending", "failed"]
        if include_no_subtitle:
            statuses.append("no_subtitle")
        placeholders = ",".join("?" for _ in statuses)
        with self.session() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT bvid
                  FROM videos
                 WHERE subtitle_status IN ({placeholders})
                UNION
                SELECT DISTINCT l.bvid
                  FROM ingest_logs l
                 WHERE l.status = 'failed'
                   AND NOT EXISTS (
                       SELECT 1 FROM videos v
                        WHERE v.bvid = l.bvid AND v.subtitle_status = 'ok'
                   )
                """,
                statuses,
            ).fetchall()
            return [str(r[0]) for r in rows if r[0]]

    def optimize_db(self) -> Dict[str, Any]:
        """大批量入库完成后压缩 FTS5 倒排段、更新统计信息并回收 WAL，保持百万级检索毫秒响应."""
        with self.session() as conn:
            conn.execute("INSERT INTO subtitles_fts(subtitles_fts) VALUES('optimize');")
            conn.execute("ANALYZE;")
        # 事务提交后再执行 WAL checkpoint，避免与未提交写事务自锁
        conn = self.connect()
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        finally:
            conn.close()
        size_mb = (
            round(self.db_path.stat().st_size / (1024 * 1024), 3)
            if self.db_path.exists()
            else 0.0
        )
        return {"status": "ok", "db_path": str(self.db_path), "db_size_mb": size_mb}

    def upsert_video(self, conn: sqlite3.Connection, rec: VideoRecord) -> None:
        extra_str = (
            rec.extra_meta
            if isinstance(rec.extra_meta, str)
            else json.dumps(rec.extra_meta, ensure_ascii=False)
        )
        conn.execute(
            """
            INSERT INTO videos (
                bvid, page, cid, up_uid, up_name, raw_up_name,
                title, part_title, pub_date, pub_ts, duration_sec,
                subtitle_lan, subtitle_status, line_count,
                scene_type, scene_confidence, scene_reason, scene_locked,
                is_visible, extra_meta
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?
            )
            ON CONFLICT(bvid, page) DO UPDATE SET
                cid              = COALESCE(excluded.cid, videos.cid),
                up_uid           = COALESCE(excluded.up_uid, videos.up_uid),
                up_name          = excluded.up_name,
                raw_up_name      = CASE WHEN excluded.raw_up_name != '' THEN excluded.raw_up_name ELSE videos.raw_up_name END,
                title            = CASE WHEN excluded.title != '' THEN excluded.title ELSE videos.title END,
                part_title       = CASE WHEN excluded.part_title != '' THEN excluded.part_title ELSE videos.part_title END,
                pub_date         = excluded.pub_date,
                pub_ts           = CASE WHEN excluded.pub_ts > 0 THEN excluded.pub_ts ELSE videos.pub_ts END,
                duration_sec     = CASE WHEN excluded.duration_sec > 0 THEN excluded.duration_sec ELSE videos.duration_sec END,
                subtitle_lan     = CASE WHEN excluded.subtitle_lan != '' THEN excluded.subtitle_lan ELSE videos.subtitle_lan END,
                subtitle_status  = CASE
                                       WHEN videos.subtitle_status = 'asr_empty' AND excluded.subtitle_status = 'no_subtitle'
                                       THEN 'asr_empty'
                                       ELSE excluded.subtitle_status
                                   END,
                line_count       = excluded.line_count,
                -- 若已有非 unknown 的锁定场景标签，增量入库不破坏它
                scene_type       = CASE
                                       WHEN videos.scene_locked = 1 THEN videos.scene_type
                                       WHEN excluded.scene_type != 'unknown' THEN excluded.scene_type
                                       ELSE videos.scene_type
                                   END,
                extra_meta       = CASE WHEN excluded.extra_meta != '{}' THEN excluded.extra_meta ELSE videos.extra_meta END,
                updated_at       = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
            """,
            (
                rec.bvid,
                rec.page,
                rec.cid,
                rec.up_uid,
                rec.up_name,
                rec.raw_up_name,
                rec.title,
                rec.part_title,
                rec.pub_date,
                rec.pub_ts,
                rec.duration_sec,
                rec.subtitle_lan,
                rec.subtitle_status,
                rec.line_count,
                rec.scene_type,
                rec.scene_confidence,
                rec.scene_reason,
                rec.scene_locked,
                rec.is_visible,
                extra_str,
            ),
        )

    def save_video_and_subtitles(
        self,
        video: VideoRecord,
        subtitles: Sequence[SubtitleRow],
        *,
        replace_existing: bool = False,
    ) -> Tuple[int, int]:
        """事务性写入视频分 P 元数据与字幕列表，返回 (新增条数, 忽略重复条数)."""
        with self.session() as conn:
            existing = conn.execute(
                "SELECT scene_type, is_visible FROM videos WHERE bvid = ? AND page = ?",
                (video.bvid, video.page),
            ).fetchone()

            effective_scene = video.scene_type
            effective_visible = video.is_visible
            if existing:
                if existing["scene_type"] != "unknown" and video.scene_type == "unknown":
                    effective_scene = existing["scene_type"]
                effective_visible = existing["is_visible"]

            if replace_existing:
                conn.execute(
                    "DELETE FROM subtitles WHERE bvid = ? AND page = ?",
                    (video.bvid, video.page),
                )

            before_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM subtitles WHERE bvid = ? AND page = ?",
                    (video.bvid, video.page),
                ).fetchone()[0]
            )
            rows_to_insert = [
                (
                    s.up_name,
                    s.bvid,
                    s.page,
                    s.pub_date,
                    round(float(s.start_sec), 2),
                    round(float(s.end_sec), 2),
                    s.content,
                    tokenize_for_fts(s.content),
                    effective_scene if s.scene_type == "unknown" else s.scene_type,
                    s.speaker_name,
                    round(float(s.speaker_conf or 0.0), 3),
                    effective_visible,
                    s.source,
                )
                for s in subtitles
            ]

            conn.executemany(
                """
                INSERT OR IGNORE INTO subtitles (
                    up_name, bvid, page, pub_date,
                    start_sec, end_sec, content, content_fts,
                    scene_type, speaker_name, speaker_conf, is_visible, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )

            total_in_db = int(
                conn.execute(
                    "SELECT COUNT(*) FROM subtitles WHERE bvid = ? AND page = ?",
                    (video.bvid, video.page),
                ).fetchone()[0]
            )
            inserted = total_in_db - before_count
            skipped = len(rows_to_insert) - inserted
            video.line_count = total_in_db
            self.upsert_video(conn, video)
            return inserted, skipped

    def log_ingest(
        self, run_id: str, bvid: str, page: int, status: str, detail: str = ""
    ) -> None:
        with self.session() as conn:
            conn.execute(
                """
                INSERT INTO ingest_logs (run_id, bvid, page, status, detail)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, bvid, page, status, detail[:1000]),
            )

    def update_scene_type(
        self,
        bvid: str,
        page: Optional[int],
        scene_type: str,
        *,
        confidence: float = 1.0,
        reason: str = "",
        lock: bool = False,
    ) -> int:
        """更新指定视频（或分 P）的 scene_type，并同步到 subtitles 表."""
        if scene_type not in VALID_SCENE_TYPES:
            raise ValueError(
                f"无效 scene_type={scene_type!r}，可选值：{VALID_SCENE_TYPES}"
            )
        with self.session() as conn:
            if page is None:
                cur = conn.execute(
                    """
                    UPDATE videos
                       SET scene_type = ?,
                           scene_confidence = ?,
                           scene_reason = ?,
                           scene_locked = CASE WHEN ? = 1 THEN 1 ELSE scene_locked END,
                           updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                     WHERE bvid = ?
                    """,
                    (scene_type, confidence, reason, 1 if lock else 0, bvid),
                )
                conn.execute(
                    "UPDATE subtitles SET scene_type = ? WHERE bvid = ?",
                    (scene_type, bvid),
                )
            else:
                cur = conn.execute(
                    """
                    UPDATE videos
                       SET scene_type = ?,
                           scene_confidence = ?,
                           scene_reason = ?,
                           scene_locked = CASE WHEN ? = 1 THEN 1 ELSE scene_locked END,
                           updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                     WHERE bvid = ? AND page = ?
                    """,
                    (scene_type, confidence, reason, 1 if lock else 0, bvid, page),
                )
                conn.execute(
                    "UPDATE subtitles SET scene_type = ? WHERE bvid = ? AND page = ?",
                    (scene_type, bvid, page),
                )
            return cur.rowcount

    def update_visibility(
        self, bvid: str, page: Optional[int], is_visible: int
    ) -> int:
        val = 1 if is_visible else 0
        with self.session() as conn:
            if page is None:
                cur = conn.execute(
                    """
                    UPDATE videos
                       SET is_visible = ?,
                           updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                     WHERE bvid = ?
                    """,
                    (val, bvid),
                )
                conn.execute(
                    "UPDATE subtitles SET is_visible = ? WHERE bvid = ?",
                    (val, bvid),
                )
            else:
                cur = conn.execute(
                    """
                    UPDATE videos
                       SET is_visible = ?,
                           updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                     WHERE bvid = ? AND page = ?
                    """,
                    (val, bvid, page),
                )
                conn.execute(
                    "UPDATE subtitles SET is_visible = ? WHERE bvid = ? AND page = ?",
                    (val, bvid, page),
                )
            return cur.rowcount

    def delete_video(
        self,
        bvid: str,
        page: Optional[int] = None,
        *,
        clear_logs: bool = True,
    ) -> Dict[str, Any]:
        """彻底删除指定视频分 P（或整个 BV 号全部分 P）及其关联字幕、FTS5 索引与抓取日志.

        典型场景：视频刚投稿尚未生成 B 站 AI 字幕时被提前抓入库（标记为 no_subtitle/pending），
        删除后可等待 AI 字幕生成完毕再重新增量抓取入库。
        """
        clean_bvid = (bvid or "").strip()
        if not clean_bvid:
            raise ValueError("bvid 不能为空")
        with self.session() as conn:
            if page is None:
                # 先删 subtitles（trg_subtitles_ad 自动同步清理 FTS5 索引），再删 videos
                s_cur = conn.execute(
                    "DELETE FROM subtitles WHERE bvid = ?",
                    (clean_bvid,),
                )
                v_cur = conn.execute(
                    "DELETE FROM videos WHERE bvid = ?",
                    (clean_bvid,),
                )
                deleted_logs = 0
                if clear_logs:
                    l_cur = conn.execute(
                        "DELETE FROM ingest_logs WHERE bvid = ?",
                        (clean_bvid,),
                    )
                    deleted_logs = max(0, int(l_cur.rowcount or 0))
            else:
                p = int(page)
                s_cur = conn.execute(
                    "DELETE FROM subtitles WHERE bvid = ? AND page = ?",
                    (clean_bvid, p),
                )
                v_cur = conn.execute(
                    "DELETE FROM videos WHERE bvid = ? AND page = ?",
                    (clean_bvid, p),
                )
                deleted_logs = 0
                if clear_logs:
                    l_cur = conn.execute(
                        "DELETE FROM ingest_logs WHERE bvid = ? AND page = ?",
                        (clean_bvid, p),
                    )
                    deleted_logs = max(0, int(l_cur.rowcount or 0))
                    # 若该 BV 号已无剩余分 P，顺手清理 page=0 的视频级日志
                    rem = conn.execute(
                        "SELECT COUNT(*) FROM videos WHERE bvid = ?",
                        (clean_bvid,),
                    ).fetchone()[0]
                    if int(rem or 0) == 0:
                        l0 = conn.execute(
                            "DELETE FROM ingest_logs WHERE bvid = ? AND page = 0",
                            (clean_bvid,),
                        )
                        deleted_logs += max(0, int(l0.rowcount or 0))

            deleted_videos = max(0, int(v_cur.rowcount or 0))
            deleted_subtitles = max(0, int(s_cur.rowcount or 0))

        return {
            "found": deleted_videos > 0,
            "bvid": clean_bvid,
            "page": int(page) if page is not None else None,
            "deleted_videos": deleted_videos,
            "deleted_subtitles": deleted_subtitles,
            "deleted_logs": deleted_logs,
        }

    def update_subtitle_speaker(
        self,
        subtitle_id: int,
        speaker_name: str,
        *,
        confidence: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """人工修正单条字幕的发言人标签，并更新所属分 P 的 updated_at 以便极速热同步上云."""
        cleaned = normalize_speaker_label(speaker_name)
        conf = float(confidence) if cleaned else 0.0
        with self.session() as conn:
            row = conn.execute(
                "SELECT id, bvid, page, speaker_name FROM subtitles WHERE id = ?",
                (int(subtitle_id),),
            ).fetchone()
            if not row:
                return None
            bvid = str(row["bvid"])
            page = int(row["page"])
            conn.execute(
                "UPDATE subtitles SET speaker_name = ?, speaker_conf = ? WHERE id = ?",
                (cleaned, conf, int(subtitle_id)),
            )
            conn.execute(
                """
                UPDATE videos
                   SET voiceprint_status = CASE WHEN ? != '' OR voiceprint_status = 'ok' THEN 'ok' ELSE voiceprint_status END,
                       updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                 WHERE bvid = ? AND page = ?
                """,
                (cleaned, bvid, page),
            )
        return {
            "id": int(subtitle_id),
            "bvid": bvid,
            "page": page,
            "old_speaker": str(row["speaker_name"] or ""),
            "speaker_name": cleaned,
            "speaker_conf": conf,
        }

    def batch_update_video_speakers(
        self,
        bvid: str,
        page: Optional[int],
        speaker_name: str,
        *,
        from_speaker: Optional[str] = None,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """按场次/分P（可选时间段与原发言人条件）批量修正字幕发言人标签."""
        cleaned = normalize_speaker_label(speaker_name)
        conf = float(confidence) if cleaned else 0.0

        where_parts = ["bvid = ?"]
        params: List[Any] = [bvid]
        if page is not None:
            where_parts.append("page = ?")
            params.append(int(page))
        if start_sec is not None:
            where_parts.append("start_sec >= ?")
            params.append(float(start_sec))
        if end_sec is not None:
            where_parts.append("start_sec <= ?")
            params.append(float(end_sec))
        if from_speaker is not None and from_speaker != "__ALL__":
            if from_speaker == "__EMPTY__":
                where_parts.append("speaker_name = ''")
            else:
                where_parts.append("speaker_name = ?")
                params.append(normalize_speaker_label(from_speaker))

        where_sql = " AND ".join(where_parts)
        with self.session() as conn:
            if page is None:
                vrows = conn.execute(
                    "SELECT page FROM videos WHERE bvid = ?", (bvid,)
                ).fetchall()
            else:
                vrows = conn.execute(
                    "SELECT page FROM videos WHERE bvid = ? AND page = ?",
                    (bvid, int(page)),
                ).fetchall()
            if not vrows:
                return {"found": False, "affected_rows": 0, "speaker_name": cleaned}

            cur = conn.execute(
                f"UPDATE subtitles SET speaker_name = ?, speaker_conf = ? WHERE {where_sql}",
                [cleaned, conf, *params],
            )
            affected = max(0, int(cur.rowcount or 0))
            if affected > 0:
                if page is None:
                    conn.execute(
                        """
                        UPDATE videos
                           SET voiceprint_status = CASE WHEN ? != '' OR voiceprint_status = 'ok' THEN 'ok' ELSE voiceprint_status END,
                               updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                         WHERE bvid = ?
                        """,
                        (cleaned, bvid),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE videos
                           SET voiceprint_status = CASE WHEN ? != '' OR voiceprint_status = 'ok' THEN 'ok' ELSE voiceprint_status END,
                               updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                         WHERE bvid = ? AND page = ?
                        """,
                        (cleaned, bvid, int(page)),
                    )
        return {
            "found": True,
            "bvid": bvid,
            "page": page,
            "affected_rows": affected,
            "speaker_name": cleaned,
            "speaker_conf": conf,
        }

    def get_stats(self) -> Dict[str, Any]:
        with self.session() as conn:
            # 全部聚合走轻量 videos 表（仅数千行，<3ms），避免在云端小内存机器上全表扫描 579 万行 subtitles
            summary = conn.execute(
                """
                SELECT COUNT(*) AS video_total,
                       COALESCE(SUM(CASE WHEN subtitle_status = 'ok' THEN 1 ELSE 0 END), 0) AS video_ok,
                       COALESCE(SUM(line_count), 0) AS sub_total,
                       COALESCE(SUM(CASE WHEN is_visible = 1 THEN line_count ELSE 0 END), 0) AS sub_visible
                  FROM videos
                """
            ).fetchone()
            video_total = int(summary["video_total"] or 0)
            video_ok = int(summary["video_ok"] or 0)
            sub_total = int(summary["sub_total"] or 0)
            sub_visible = int(summary["sub_visible"] or 0)

            by_up = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT up_name,
                           COUNT(*) AS video_parts,
                           SUM(CASE WHEN subtitle_status = 'ok' THEN 1 ELSE 0 END) AS ok_parts,
                           COALESCE(SUM(line_count), 0) AS subtitle_lines
                      FROM videos
                     GROUP BY up_name
                     ORDER BY subtitle_lines DESC, up_name ASC
                    """
                ).fetchall()
            ]
            by_scene = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT scene_type,
                           COUNT(*) AS video_parts,
                           COALESCE(SUM(line_count), 0) AS subtitle_lines
                      FROM videos
                     GROUP BY scene_type
                     ORDER BY video_parts DESC
                    """
                ).fetchall()
            ]
            date_range = conn.execute(
                """
                SELECT MIN(pub_date) AS min_date,
                       MAX(pub_date) AS max_date
                  FROM videos
                 WHERE subtitle_status = 'ok'
                   AND is_visible = 1
                   AND pub_date != ''
                """
            ).fetchone()
            if not date_range or not date_range["min_date"]:
                date_range = conn.execute(
                    "SELECT MIN(pub_date) AS min_date, MAX(pub_date) AS max_date "
                    "FROM videos WHERE pub_date != ''"
                ).fetchone()
            latest_updated = conn.execute(
                "SELECT MAX(updated_at) AS latest_updated_at FROM videos"
            ).fetchone()
            latest_video_row = conn.execute(
                """
                SELECT bvid, page, up_name, title, pub_date, pub_ts
                  FROM videos
                 WHERE subtitle_status = 'ok'
                   AND is_visible = 1
                   AND pub_date != ''
                 ORDER BY pub_date DESC, pub_ts DESC, updated_at DESC, page ASC
                 LIMIT 1
                """
            ).fetchone()

        latest_video: Optional[Dict[str, Any]] = None
        if latest_video_row:
            lv_bvid = str(latest_video_row["bvid"] or "")
            lv_page = int(latest_video_row["page"] or 1)
            lv_url = (
                f"https://www.bilibili.com/video/{lv_bvid}/?p={lv_page}"
                if lv_page > 1
                else f"https://www.bilibili.com/video/{lv_bvid}/"
            )
            latest_video = {
                "bvid": lv_bvid,
                "page": lv_page,
                "up_name": str(latest_video_row["up_name"] or ""),
                "title": str(latest_video_row["title"] or ""),
                "pub_date": str(latest_video_row["pub_date"] or ""),
                "video_url": lv_url,
            }

        db_size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "db_path": str(self.db_path),
            "db_size_mb": round(db_size_bytes / (1024 * 1024), 3),
            "video_parts_total": video_total,
            "video_parts_ok": video_ok,
            "subtitles_total": sub_total,
            "subtitles_visible": sub_visible,
            "date_min": date_range["min_date"] if date_range else None,
            "date_max": date_range["max_date"] if date_range else None,
            "latest_updated_at": (
                latest_updated["latest_updated_at"] if latest_updated else None
            ),
            "latest_video": latest_video,
            "by_up": by_up,
            "by_scene": by_scene,
        }

    def rebuild_fts(self) -> None:
        """如手动修改过字幕或迁移导入数据，可一键重建 FTS5 索引."""
        with self.session() as conn:
            conn.execute("INSERT INTO subtitles_fts(subtitles_fts) VALUES('rebuild');")
