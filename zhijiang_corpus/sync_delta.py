"""增量同步引擎：比对本地与云端数据库清单，仅打包传输新增或变动的分 P（通常仅几百 KB ~ 几 MB），
在云端 SQLite WAL 模式下热写入并自动触发 FTS5 增量索引，零停服、零全量覆盖。"""

from __future__ import annotations

import gzip
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .db import CorpusDB
from .tokenizer import tokenize_for_fts

VIDEO_COLS = (
    "bvid",
    "page",
    "cid",
    "up_uid",
    "up_name",
    "raw_up_name",
    "title",
    "part_title",
    "pub_date",
    "pub_ts",
    "duration_sec",
    "subtitle_lan",
    "subtitle_status",
    "line_count",
    "scene_type",
    "scene_confidence",
    "scene_reason",
    "scene_locked",
    "voiceprint_status",
    "is_visible",
    "extra_meta",
    "created_at",
    "updated_at",
)


def _part_key(bvid: str, page: int) -> str:
    return f"{bvid}#{int(page)}"


def build_manifest(db: CorpusDB) -> Dict[str, Dict[str, Any]]:
    """导出当前数据库所有分 P 的轻量签名清单（3,300+ 行耗时 <5ms，约 50KB）."""
    db.init_db()
    manifest: Dict[str, Dict[str, Any]] = {}
    with db.session() as conn:
        rows = conn.execute(
            """
            SELECT bvid, page, updated_at, line_count,
                   subtitle_status, scene_type, voiceprint_status, is_visible
              FROM videos
            """
        ).fetchall()
        for r in rows:
            manifest[_part_key(r["bvid"], int(r["page"]))] = {
                "updated_at": str(r["updated_at"] or ""),
                "line_count": int(r["line_count"] or 0),
                "subtitle_status": str(r["subtitle_status"] or ""),
                "scene_type": str(r["scene_type"] or ""),
                "voiceprint_status": str(r["voiceprint_status"] or ""),
                "is_visible": int(r["is_visible"] if r["is_visible"] is not None else 1),
            }
    return manifest


def _is_part_changed(local_entry: Dict[str, Any], remote_entry: Dict[str, Any] | None) -> bool:
    if remote_entry is None:
        return True
    for field in ("line_count", "subtitle_status", "scene_type", "voiceprint_status", "is_visible"):
        if local_entry.get(field) != remote_entry.get(field):
            return True
    if str(local_entry.get("updated_at") or "") > str(remote_entry.get("updated_at") or ""):
        return True
    return False


def export_delta(
    db: CorpusDB,
    remote_manifest: Dict[str, Dict[str, Any]],
    out_path: str | Path,
    *,
    include_deletes: bool = True,
) -> Dict[str, Any]:
    """对比本地库与云端清单，导出新增/变动分 P 及本地已删除分 P 到 gzip JSON 补丁包."""
    local_manifest = build_manifest(db)
    changed_keys: List[Tuple[str, int]] = []
    new_parts = 0
    updated_parts = 0

    for key, loc in local_manifest.items():
        rem = remote_manifest.get(key)
        if _is_part_changed(loc, rem):
            bvid, page_str = key.rsplit("#", 1)
            changed_keys.append((bvid, int(page_str)))
            if rem is None:
                new_parts += 1
            else:
                updated_parts += 1

    # 检测本地已删除但云端仍存在的分 P（带防误清库保护：本地为空或单次删除比例过高时不自动删除）
    deleted_keys: List[Tuple[str, int]] = []
    if include_deletes and remote_manifest and local_manifest:
        raw_deleted: List[Tuple[str, int]] = []
        for key in remote_manifest:
            if key not in local_manifest and "#" in key:
                bvid, page_str = key.rsplit("#", 1)
                raw_deleted.append((bvid, int(page_str)))
        if len(remote_manifest) <= 20 or len(raw_deleted) <= max(20, int(len(remote_manifest) * 0.2)):
            deleted_keys = sorted(raw_deleted)

    changed_keys.sort()
    parts_payload: List[Dict[str, Any]] = []
    total_subtitles = 0

    if changed_keys:
        cols_sql = ", ".join(VIDEO_COLS)
        with db.session() as conn:
            for bvid, page in changed_keys:
                vrow = conn.execute(
                    f"SELECT {cols_sql} FROM videos WHERE bvid = ? AND page = ?",
                    (bvid, page),
                ).fetchone()
                if not vrow:
                    continue
                srows = conn.execute(
                    """
                    SELECT start_sec, end_sec, content, scene_type, is_visible, source,
                           speaker_name, speaker_conf
                      FROM subtitles
                     WHERE bvid = ? AND page = ?
                     ORDER BY start_sec ASC, id ASC
                    """,
                    (bvid, page),
                ).fetchall()
                subs_compact = [
                    [
                        round(float(s["start_sec"]), 2),
                        round(float(s["end_sec"]), 2),
                        str(s["content"]),
                        str(s["scene_type"]),
                        int(s["is_visible"]),
                        str(s["source"]),
                        str(s["speaker_name"] or ""),
                        round(float(s["speaker_conf"] or 0.0), 3),
                    ]
                    for s in srows
                ]
                total_subtitles += len(subs_compact)
                parts_payload.append(
                    {
                        "video": {col: vrow[col] for col in VIDEO_COLS},
                        "subtitles": subs_compact,
                    }
                )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    deleted_payload = [[bvid, page] for bvid, page in deleted_keys]
    payload = {
        "version": 1,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parts_count": len(parts_payload),
        "deleted_count": len(deleted_payload),
        "subtitles_count": total_subtitles,
        "parts": parts_payload,
        "deleted_parts": deleted_payload,
    }
    raw_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(out, "wb", compresslevel=6) as gz:
        gz.write(raw_bytes)

    size_kb = round(out.stat().st_size / 1024.0, 2)
    return {
        "status": "ok",
        "out_path": str(out),
        "local_parts": len(local_manifest),
        "remote_parts": len(remote_manifest),
        "parts_changed": len(parts_payload),
        "parts_new": new_parts,
        "parts_updated": updated_parts,
        "parts_deleted": len(deleted_payload),
        "subtitles_exported": total_subtitles,
        "delta_size_kb": size_kb,
    }


def apply_delta(
    db: CorpusDB,
    delta_path: str | Path,
    *,
    optimize: bool = False,
) -> Dict[str, Any]:
    """在云端数据库上热应用增量补丁包（SQLite WAL 事务 + FTS5 触发器自动同步，零停服）."""
    db.init_db()
    path = Path(delta_path)
    if not path.exists():
        raise FileNotFoundError(f"增量补丁文件不存在: {path}")

    with gzip.open(path, "rb") as gz:
        payload = json.loads(gz.read().decode("utf-8"))

    parts = payload.get("parts") or []
    deleted_parts = payload.get("deleted_parts") or []
    parts_applied = 0
    subtitles_inserted = 0
    parts_deleted = 0
    subtitles_deleted = 0

    if parts or deleted_parts:
        placeholders = ", ".join("?" for _ in VIDEO_COLS)
        cols_sql = ", ".join(VIDEO_COLS)
        update_assigns = ", ".join(
            f"{c} = excluded.{c}" for c in VIDEO_COLS if c not in ("bvid", "page", "created_at")
        )
        upsert_sql = f"""
            INSERT INTO videos ({cols_sql})
            VALUES ({placeholders})
            ON CONFLICT(bvid, page) DO UPDATE SET
                {update_assigns}
        """

        with db.session() as conn:
            for d_item in deleted_parts:
                if not isinstance(d_item, (list, tuple)) or len(d_item) < 2:
                    continue
                d_bvid = str(d_item[0]).strip()
                d_page = int(d_item[1])
                if not d_bvid:
                    continue
                sc = conn.execute(
                    "DELETE FROM subtitles WHERE bvid = ? AND page = ?",
                    (d_bvid, d_page),
                )
                vc = conn.execute(
                    "DELETE FROM videos WHERE bvid = ? AND page = ?",
                    (d_bvid, d_page),
                )
                conn.execute(
                    "DELETE FROM ingest_logs WHERE bvid = ? AND page = ?",
                    (d_bvid, d_page),
                )
                parts_deleted += max(0, int(vc.rowcount or 0))
                subtitles_deleted += max(0, int(sc.rowcount or 0))

            for item in parts:
                v = item["video"]
                bvid = str(v["bvid"])
                page = int(v["page"])
                up_name = str(v["up_name"])
                pub_date = str(v["pub_date"])
                subs = item.get("subtitles") or []

                # 先删旧字幕（trg_subtitles_ad 自动清理旧 FTS5 条目），再批量写入新字幕（trg_subtitles_ai 自动建索引）
                conn.execute(
                    "DELETE FROM subtitles WHERE bvid = ? AND page = ?",
                    (bvid, page),
                )
                if subs:
                    rows_to_insert = [
                        (
                            up_name,
                            bvid,
                            page,
                            pub_date,
                            round(float(s[0]), 2),
                            round(float(s[1]), 2),
                            str(s[2]),
                            tokenize_for_fts(str(s[2])),
                            str(s[3]),
                            str(s[6]) if len(s) > 6 else "",
                            round(float(s[7]), 3) if len(s) > 7 else 0.0,
                            int(s[4]),
                            str(s[5]),
                        )
                        for s in subs
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
                    actual_count = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM subtitles WHERE bvid = ? AND page = ?",
                            (bvid, page),
                        ).fetchone()[0]
                    )
                else:
                    actual_count = 0

                v["line_count"] = actual_count
                if "voiceprint_status" not in v or v["voiceprint_status"] is None:
                    v["voiceprint_status"] = ""
                conn.execute(upsert_sql, [v.get(c) for c in VIDEO_COLS])
                parts_applied += 1
                subtitles_inserted += actual_count

        # 默认不自动触发整库 2GB FTS5 全量 optimize（避免云端 2C2G 磁盘卡顿 1~3 分钟）；
        # 仅执行非阻塞 PASSIVE checkpoint，立即对公网读请求生效且零停服
        if optimize:
            with db.session() as conn:
                conn.execute("INSERT INTO subtitles_fts(subtitles_fts) VALUES('optimize');")
                conn.execute("ANALYZE;")

        conn = db.connect()
        try:
            conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
        finally:
            conn.close()

    st = db.get_stats()
    return {
        "status": "ok",
        "parts_applied": parts_applied,
        "parts_deleted": parts_deleted,
        "subtitles_inserted": subtitles_inserted,
        "subtitles_deleted": subtitles_deleted,
        "live_stats": {
            "subtitles_total": st["subtitles_total"],
            "video_parts_ok": st["video_parts_ok"],
            "date_min": st["date_min"],
            "date_max": st["date_max"],
            "latest_updated_at": st.get("latest_updated_at"),
            "db_size_mb": st["db_size_mb"],
        },
    }


def export_scene_tags(
    db: CorpusDB,
    out_path: str | Path,
    *,
    only_locked: bool = True,
) -> Dict[str, Any]:
    """导出纯场景标签与可见性轻量补丁（仅数 KB，不携带任何字幕文本）."""
    where_sql = "WHERE scene_locked = 1" if only_locked else ""
    with db.session() as conn:
        rows = conn.execute(
            f"""
            SELECT bvid, page, scene_type, scene_confidence,
                   scene_reason, scene_locked, is_visible, updated_at
              FROM videos
            {where_sql}
             ORDER BY bvid ASC, page ASC
            """
        ).fetchall()

    items = [
        [
            str(r["bvid"]),
            int(r["page"]),
            str(r["scene_type"] or "unknown"),
            float(r["scene_confidence"] or 0.0),
            str(r["scene_reason"] or ""),
            int(r["scene_locked"] or 0),
            int(r["is_visible"] if r["is_visible"] is not None else 1),
            str(r["updated_at"] or ""),
        ]
        for r in rows
    ]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "type": "scene_tags",
        "only_locked": only_locked,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(items),
        "items": items,
    }
    raw_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    out.write_bytes(gzip.compress(raw_bytes, compresslevel=6))
    return {
        "only_locked": only_locked,
        "parts_exported": len(items),
        "bvids_exported": len({it[0] for it in items}),
        "patch_size_kb": round(out.stat().st_size / 1024.0, 2),
        "out_path": str(out),
    }


def apply_scene_tags(db: CorpusDB, patch_path: str | Path) -> Dict[str, Any]:
    """在云端热更新场景标签与可见性（仅改有差异的行，不触碰字幕文本与 FTS5 索引）."""
    p = Path(patch_path)
    if not p.exists():
        raise FileNotFoundError(f"找不到标签补丁文件: {p}")

    raw = p.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    payload = json.loads(raw.decode("utf-8"))
    items = payload.get("items") or []

    parts_updated = 0
    subtitles_updated = 0
    bvids_touched: set[str] = set()

    if items:
        with db.session() as conn:
            for it in items:
                bvid = str(it[0])
                page = int(it[1])
                scene_type = str(it[2])
                scene_conf = float(it[3])
                scene_reason = str(it[4])
                scene_locked = int(it[5])
                is_visible = int(it[6])
                updated_at = str(it[7]) if len(it) > 7 and it[7] else ""

                cur = conn.execute(
                    """
                    SELECT scene_type, scene_locked, is_visible
                      FROM videos
                     WHERE bvid = ? AND page = ?
                    """,
                    (bvid, page),
                ).fetchone()
                if not cur:
                    continue
                if (
                    str(cur["scene_type"] or "") == scene_type
                    and int(cur["scene_locked"] or 0) == scene_locked
                    and int(cur["is_visible"] if cur["is_visible"] is not None else 1) == is_visible
                ):
                    continue

                conn.execute(
                    """
                    UPDATE videos
                       SET scene_type = ?,
                           scene_confidence = ?,
                           scene_reason = ?,
                           scene_locked = ?,
                           is_visible = ?,
                           updated_at = CASE WHEN ? != '' THEN ? ELSE strftime('%Y-%m-%dT%H:%M:%SZ', 'now') END
                     WHERE bvid = ? AND page = ?
                    """,
                    (
                        scene_type,
                        scene_conf,
                        scene_reason,
                        scene_locked,
                        is_visible,
                        updated_at,
                        updated_at,
                        bvid,
                        page,
                    ),
                )
                scur = conn.execute(
                    """
                    UPDATE subtitles
                       SET scene_type = ?, is_visible = ?
                     WHERE bvid = ? AND page = ?
                    """,
                    (scene_type, is_visible, bvid, page),
                )
                parts_updated += 1
                subtitles_updated += max(0, int(scur.rowcount or 0))
                bvids_touched.add(bvid)

        conn = db.connect()
        try:
            conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
        finally:
            conn.close()

    st = db.get_stats()
    return {
        "status": "ok",
        "parts_scanned": len(items),
        "parts_updated": parts_updated,
        "bvids_updated": len(bvids_touched),
        "subtitles_updated": subtitles_updated,
        "live_stats": {
            "subtitles_total": st["subtitles_total"],
            "video_parts_ok": st["video_parts_ok"],
            "date_min": st["date_min"],
            "date_max": st["date_max"],
            "latest_updated_at": st.get("latest_updated_at"),
        },
    }


def export_speakers(
    db: CorpusDB,
    out_path: str | Path,
    *,
    remote_manifest: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """导出纯声纹发言人标注轻量补丁（不含字幕文本、零 FTS5 重建，仅携带起止秒数与 speaker_name/conf）."""
    db.init_db()
    with db.session() as conn:
        parts = conn.execute(
            """
            SELECT bvid, page, updated_at
              FROM videos
             WHERE voiceprint_status = 'ok'
             ORDER BY bvid ASC, page ASC
            """
        ).fetchall()

        payload_parts: List[Dict[str, Any]] = []
        lines_total = 0
        for p in parts:
            bvid = str(p["bvid"])
            page = int(p["page"])
            local_upd = str(p["updated_at"] or "")
            if remote_manifest is not None:
                rem = remote_manifest.get(_part_key(bvid, page))
                if (
                    rem
                    and str(rem.get("voiceprint_status") or "") == "ok"
                    and local_upd <= str(rem.get("updated_at") or "")
                ):
                    continue
            srows = conn.execute(
                """
                SELECT start_sec, end_sec, speaker_name, speaker_conf
                  FROM subtitles
                 WHERE bvid = ? AND page = ? AND speaker_name != ''
                 ORDER BY start_sec ASC, id ASC
                """,
                (bvid, page),
            ).fetchall()
            compact = [
                [
                    round(float(r["start_sec"]), 2),
                    round(float(r["end_sec"]), 2),
                    str(r["speaker_name"]),
                    round(float(r["speaker_conf"] or 0.0), 3),
                ]
                for r in srows
            ]
            lines_total += len(compact)
            payload_parts.append(
                {
                    "bvid": bvid,
                    "page": page,
                    "updated_at": local_upd,
                    "reset_first": True,
                    "rows": compact,
                }
            )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "type": "speakers",
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parts_count": len(payload_parts),
        "lines_count": lines_total,
        "parts": payload_parts,
    }
    raw_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    out.write_bytes(gzip.compress(raw_bytes, compresslevel=6))
    return {
        "parts_exported": len(payload_parts),
        "lines_exported": lines_total,
        "patch_size_kb": round(out.stat().st_size / 1024.0, 2),
        "out_path": str(out),
    }


def apply_speakers(db: CorpusDB, patch_path: str | Path) -> Dict[str, Any]:
    """在云端 WAL 模式下极速热更新声纹发言人标注（不触发 FTS5 索引重写，零停服）."""
    db.init_db()
    p = Path(patch_path)
    if not p.exists():
        raise FileNotFoundError(f"找不到声纹补丁文件: {p}")

    raw = p.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    payload = json.loads(raw.decode("utf-8"))
    parts = payload.get("parts") or []

    parts_applied = 0
    lines_updated = 0

    if parts:
        with db.session() as conn:
            for item in parts:
                bvid = str(item["bvid"])
                page = int(item["page"])
                rows = item.get("rows") or []
                part_upd = str(item.get("updated_at") or "")
                if item.get("reset_first"):
                    conn.execute(
                        "UPDATE subtitles SET speaker_name = '', speaker_conf = 0.0 WHERE bvid = ? AND page = ?",
                        (bvid, page),
                    )
                if rows:
                    conn.executemany(
                        """
                        UPDATE subtitles
                           SET speaker_name = ?, speaker_conf = ?
                         WHERE bvid = ? AND page = ? AND start_sec = ? AND end_sec = ?
                        """,
                        [
                            (str(r[2]), round(float(r[3]), 3), bvid, page, round(float(r[0]), 2), round(float(r[1]), 2))
                            for r in rows
                        ],
                    )
                    lines_updated += len(rows)
                conn.execute(
                    """
                    UPDATE videos
                       SET voiceprint_status = 'ok',
                           updated_at = CASE WHEN ? != '' THEN ? ELSE strftime('%Y-%m-%dT%H:%M:%SZ', 'now') END
                     WHERE bvid = ? AND page = ?
                    """,
                    (part_upd, part_upd, bvid, page),
                )
                parts_applied += 1

        conn = db.connect()
        try:
            conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
        finally:
            conn.close()

    return {
        "status": "ok",
        "parts_applied": parts_applied,
        "lines_updated": lines_updated,
    }
