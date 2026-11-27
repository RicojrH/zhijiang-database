"""关键词检索与上下文浏览服务（基于 SQLite FTS5 + 复合过滤索引）."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .db import CorpusDB
from .normalizer import build_fuzzy_fts_query


@dataclass
class SearchQuery:
    q: str = ""
    up_names: Optional[Sequence[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    scene_types: Optional[Sequence[str]] = None
    bvid: Optional[str] = None
    include_hidden: bool = False
    fuzzy: bool = True
    speaker_filter: bool = False
    context_lines: int = 0
    clip_pad_before: float = 1.5
    clip_pad_after: float = 1.5
    sort: str = "date_desc"  # date_desc | date_asc | relevance
    limit: int = 20
    offset: int = 0


def build_bilibili_video_url(bvid: str, page: int = 1) -> str:
    """生成规范的 B 站视频播放页链接（带尾部斜杠避免 301 重定向）."""
    if page and int(page) > 1:
        return f"https://www.bilibili.com/video/{bvid}/?p={int(page)}"
    return f"https://www.bilibili.com/video/{bvid}/"


def build_bilibili_jump_url(bvid: str, page: int, start_sec: float) -> str:
    """生成带时间戳的 B 站空降链接.

    细节修正：
      - 带尾部斜杠 /video/{bvid}/ 避免触发 B 站 301 跳转丢失参数；
      - 单 P（page == 1）省略冗余 p=1，降低播放器忽略 t= 参数的概率；
      - t 参数严格取非负整数秒。
    """
    sec = max(0, int(start_sec))
    if page and int(page) > 1:
        return f"https://www.bilibili.com/video/{bvid}/?p={int(page)}&t={sec}"
    return f"https://www.bilibili.com/video/{bvid}/?t={sec}"


def format_sec_hms(sec: float) -> str:
    total = max(0.0, float(sec))
    h = int(total // 3600)
    m = int((total % 3600) // 60)
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


class SearchService:
    """语料检索核心类，同时被 CLI 和 FastAPI 接口复用."""

    def __init__(self, db: CorpusDB):
        self.db = db

    def search(self, req: SearchQuery) -> Dict[str, Any]:
        limit = max(1, min(200, int(req.limit)))
        offset = max(0, int(req.offset))
        ctx_n = max(0, min(10, int(req.context_lines)))

        fts_expr, fuzzy_expanded = build_fuzzy_fts_query(req.q, fuzzy=req.fuzzy)
        has_q = bool(req.q and req.q.strip())

        where_clauses: List[str] = []
        params: List[Any] = []

        if not req.include_hidden:
            where_clauses.append("s.is_visible = 1")

        if req.up_names:
            cleaned_ups = [u.strip() for u in req.up_names if u and u.strip()]
            if cleaned_ups:
                placeholders = ",".join("?" for _ in cleaned_ups)
                if req.speaker_filter:
                    # 声纹发言人精准筛选：
                    # 1) 已做声纹标注的行：按实际发言人匹配（含单人及 A+B 双人重叠/换人，且跨频道捞出本人发言）
                    # 2) 未做声纹标注的纯单播行：平滑回退至录播频道 up_name
                    sp_checks = []
                    for u in cleaned_ups:
                        sp_checks.append(
                            "(s.speaker_name = ? OR s.speaker_name LIKE ? OR s.speaker_name LIKE ?)"
                        )
                        params.extend([u, f"{u}+%", f"%+{u}"])
                    where_clauses.append(
                        f"(({' OR '.join(sp_checks)}) OR (s.speaker_name = '' AND s.scene_type = 'solo' AND s.up_name IN ({placeholders})))"
                    )
                    params.extend(cleaned_ups)
                else:
                    where_clauses.append(f"s.up_name IN ({placeholders})")
                    params.extend(cleaned_ups)

        if req.speaker_filter:
            where_clauses.append("s.speaker_name != '其他/原声'")

        if req.date_from:
            where_clauses.append("s.pub_date >= ?")
            params.append(req.date_from.strip())

        if req.date_to:
            where_clauses.append("s.pub_date <= ?")
            params.append(req.date_to.strip())

        if req.scene_types:
            cleaned_scenes = [sc.strip() for sc in req.scene_types if sc and sc.strip()]
            if cleaned_scenes:
                placeholders = ",".join("?" for _ in cleaned_scenes)
                where_clauses.append(f"s.scene_type IN ({placeholders})")
                params.extend(cleaned_scenes)

        if req.bvid:
            where_clauses.append("s.bvid = ?")
            params.append(req.bvid.strip())

        # 用户输入了纯标点等无法转为 FTS token 的字符串时，回退 LIKE 匹配
        use_fts = bool(fts_expr)
        if has_q and not use_fts:
            where_clauses.append("s.content LIKE ?")
            params.append(f"%{req.q.strip()}%")

        if use_fts:
            # 使用 CROSS JOIN 强制 SQLite 先走 FTS5 倒排索引再按主键回表，
            # 避免过滤 up_name / pub_date 时错误退化为全表扫描 + 循环探查 FTS（实测提速 100 倍以上）
            from_sql = """
                FROM subtitles_fts f
                CROSS JOIN subtitles s ON s.id = f.rowid
                LEFT JOIN videos v ON v.bvid = s.bvid AND v.page = s.page
            """
            where_clauses.insert(0, "subtitles_fts MATCH ?")
            params.insert(0, fts_expr)
            if req.sort == "relevance":
                order_sql = "ORDER BY bm25(subtitles_fts) ASC, s.pub_date DESC, s.start_sec ASC"
            elif req.sort == "date_asc":
                order_sql = "ORDER BY s.pub_date ASC, s.bvid ASC, s.page ASC, s.start_sec ASC"
            else:
                order_sql = "ORDER BY s.pub_date DESC, s.bvid DESC, s.page ASC, s.start_sec ASC"
        else:
            from_sql = """
                FROM subtitles s
                LEFT JOIN videos v ON v.bvid = s.bvid AND v.page = s.page
            """
            if req.sort == "date_asc":
                order_sql = "ORDER BY s.pub_date ASC, s.bvid ASC, s.page ASC, s.start_sec ASC"
            else:
                order_sql = "ORDER BY s.pub_date DESC, s.bvid DESC, s.page ASC, s.start_sec ASC"

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        count_sql = f"SELECT COUNT(*) {from_sql} {where_sql}"
        data_sql = f"""
            SELECT
                s.id,
                s.up_name,
                s.bvid,
                s.page,
                s.pub_date,
                s.start_sec,
                s.end_sec,
                s.content,
                s.scene_type,
                s.speaker_name,
                s.speaker_conf,
                s.is_visible,
                s.source,
                COALESCE(v.title, '') AS title,
                COALESCE(v.part_title, '') AS part_title,
                COALESCE(v.duration_sec, 0) AS video_duration_sec
            {from_sql}
            {where_sql}
            {order_sql}
            LIMIT ? OFFSET ?
        """

        with self.db.session() as conn:
            total = int(conn.execute(count_sql, params).fetchone()[0])
            rows = conn.execute(data_sql, [*params, limit, offset]).fetchall()

            items: List[Dict[str, Any]] = []
            for r in rows:
                start_sec = float(r["start_sec"])
                end_sec = float(r["end_sec"])
                vid_dur = int(r["video_duration_sec"] or 0)
                clip_start = round(max(0.0, start_sec - req.clip_pad_before), 2)
                clip_end = round(end_sec + req.clip_pad_after, 2)
                if vid_dur > 0:
                    clip_end = min(float(vid_dur), clip_end)

                item: Dict[str, Any] = {
                    "id": int(r["id"]),
                    "up_name": r["up_name"],
                    "bvid": r["bvid"],
                    "page": int(r["page"]),
                    "pub_date": r["pub_date"],
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "start_hms": format_sec_hms(start_sec),
                    "end_hms": format_sec_hms(end_sec),
                    "content": r["content"],
                    "scene_type": r["scene_type"],
                    "speaker_name": str(r["speaker_name"] or ""),
                    "speaker_conf": round(float(r["speaker_conf"] or 0.0), 3),
                    "is_visible": int(r["is_visible"]),
                    "title": r["title"],
                    "part_title": r["part_title"],
                    "video_url": build_bilibili_video_url(r["bvid"], int(r["page"])),
                    "jump_url": build_bilibili_jump_url(r["bvid"], int(r["page"]), start_sec),
                    "jump_sec": max(0, int(start_sec)),
                    "clip_hint": {
                        "clip_start_sec": clip_start,
                        "clip_end_sec": clip_end,
                    },
                }

                if ctx_n > 0:
                    before = conn.execute(
                        """
                        SELECT id, start_sec, end_sec, content
                          FROM subtitles
                         WHERE bvid = ? AND page = ? AND start_sec <= ? AND id != ?
                         ORDER BY start_sec DESC, id DESC
                         LIMIT ?
                        """,
                        (r["bvid"], r["page"], start_sec, r["id"], ctx_n),
                    ).fetchall()
                    after = conn.execute(
                        """
                        SELECT id, start_sec, end_sec, content
                          FROM subtitles
                         WHERE bvid = ? AND page = ? AND start_sec >= ? AND id != ?
                         ORDER BY start_sec ASC, id ASC
                         LIMIT ?
                        """,
                        (r["bvid"], r["page"], start_sec, r["id"], ctx_n),
                    ).fetchall()
                    item["context_before"] = [dict(x) for x in reversed(before)]
                    item["context_after"] = [dict(x) for x in after]

                items.append(item)

        return {
            "query": req.q,
            "fts_expr": fts_expr,
            "fuzzy": req.fuzzy,
            "fuzzy_expanded": fuzzy_expanded,
            "filters": {
                "up_names": list(req.up_names) if req.up_names else None,
                "date_from": req.date_from,
                "date_to": req.date_to,
                "scene_types": list(req.scene_types) if req.scene_types else None,
                "bvid": req.bvid,
                "include_hidden": req.include_hidden,
            },
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "page": (offset // limit) + 1 if limit > 0 else 1,
                "total_pages": ((total + limit - 1) // limit) if (total > 0 and limit > 0) else 0,
                "has_more": (offset + len(items)) < total,
            },
            "items": items,
        }

    def list_videos(
        self,
        *,
        up_name: Optional[str] = None,
        scene_type: Optional[str] = None,
        status: Optional[str] = None,
        include_hidden: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        clauses: List[str] = []
        params: List[Any] = []
        if not include_hidden:
            clauses.append("is_visible = 1")
        if up_name:
            clauses.append("up_name = ?")
            params.append(up_name.strip())
        if scene_type:
            clauses.append("scene_type = ?")
            params.append(scene_type.strip())
        if status:
            clauses.append("subtitle_status = ?")
            params.append(status.strip())
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self.db.session() as conn:
            total = int(
                conn.execute(f"SELECT COUNT(*) FROM videos {where_sql}", params).fetchone()[0]
            )
            rows = conn.execute(
                f"""
                SELECT bvid, page, up_name, title, part_title, pub_date,
                       duration_sec, subtitle_lan, subtitle_status, line_count,
                       scene_type, scene_confidence, scene_locked, is_visible
                  FROM videos
                {where_sql}
                 ORDER BY pub_date DESC, bvid DESC, page ASC
                 LIMIT ? OFFSET ?
                """,
                [*params, max(1, min(200, limit)), max(0, offset)],
            ).fetchall()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": [dict(r) for r in rows],
        }

    def get_subtitle_window(
        self,
        bvid: str,
        page: int = 1,
        *,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        """读取指定视频时间窗口内的连续字幕（供剪辑师微调切片入出点）."""
        clauses = ["bvid = ?", "page = ?"]
        params: List[Any] = [bvid, page]
        if start_sec is not None:
            clauses.append("end_sec >= ?")
            params.append(float(start_sec))
        if end_sec is not None:
            clauses.append("start_sec <= ?")
            params.append(float(end_sec))

        with self.db.session() as conn:
            vid = conn.execute(
                "SELECT bvid, page, up_name, title, part_title, pub_date, duration_sec, scene_type "
                "FROM videos WHERE bvid = ? AND page = ?",
                (bvid, page),
            ).fetchone()
            rows = conn.execute(
                f"""
                SELECT id, start_sec, end_sec, content, scene_type, speaker_name, speaker_conf, is_visible
                  FROM subtitles
                 WHERE {' AND '.join(clauses)}
                 ORDER BY start_sec ASC, id ASC
                 LIMIT ?
                """,
                [*params, max(1, min(1000, limit))],
            ).fetchall()

        return {
            "video": dict(vid) if vid else {"bvid": bvid, "page": page},
            "count": len(rows),
            "items": [dict(r) for r in rows],
        }
