"""公网只读检索 API（部署至云端对外开放时仅挂载本路由，无任何写操作或 Cookie 暴露）."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, Request, Response

from zhijiang_corpus.config import AppConfig, PROJECT_ROOT, VALID_SCENE_TYPES
from zhijiang_corpus.db import CorpusDB
from zhijiang_corpus.metrics import SiteMetricsDB
from zhijiang_corpus.search import SearchQuery, SearchService


def _load_announcements() -> List[Dict[str, Any]]:
    """实时读取 config/announcements.json 中的有效公告，改文件即刻生效无需重启服务."""
    env_path = os.getenv("ZJ_ANNOUNCEMENTS_PATH", "").strip()
    path = Path(env_path) if env_path else (PROJECT_ROOT / "config" / "announcements.json")
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = raw.get("items", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    active_items: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("active", True) is False:
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        level = str(item.get("level", "info")).strip().lower()
        if level not in ("info", "change", "warn"):
            level = "info"
        active_items.append(
            {
                "id": str(item.get("id", "")).strip(),
                "tag": str(item.get("tag", "公告")).strip() or "公告",
                "level": level,
                "date": str(item.get("date", "")).strip(),
                "pinned": bool(item.get("pinned", False)),
                "content": content,
            }
        )
    # 置顶优先，同级按日期倒序
    pinned = sorted([x for x in active_items if x["pinned"]], key=lambda x: x["date"], reverse=True)
    normal = sorted([x for x in active_items if not x["pinned"]], key=lambda x: x["date"], reverse=True)
    return pinned + normal


def _extract_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "").strip()
    if xff:
        return xff.split(",")[0].strip()
    xri = request.headers.get("x-real-ip", "").strip()
    if xri:
        return xri
    if request.client and request.client.host:
        return str(request.client.host)
    return "unknown"


def build_public_router(db: CorpusDB, cfg: AppConfig) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["Public Read-Only API"])
    search_svc = SearchService(db)
    metrics_db = SiteMetricsDB(db.db_path.parent / "site_metrics.db")

    @router.get("/health", summary="服务健康检查")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "mode": "read_only_api"}

    @router.get("/members", summary="获取成员应援色、粉丝名与场景枚举元数据（供前端渲染）")
    def list_members() -> Dict[str, Any]:
        return {
            "members": [
                {
                    "up_name": m.up_name,
                    "uid": m.uid,
                    "official_name": m.official_name,
                    "group": m.group,
                    "fan_name": m.fan_name,
                    "color": m.color,
                }
                for m in cfg.members
            ],
            "scene_types": list(VALID_SCENE_TYPES),
            "site": {
                "author_name": os.getenv("ZJ_AUTHOR_NAME", "没品人类之一"),
                "author_bilibili_url": os.getenv(
                    "ZJ_AUTHOR_BILI_URL", "https://space.bilibili.com/396896138"
                ),
                "github_url": os.getenv("ZJ_GITHUB_URL", ""),
            },
        }

    @router.get("/announcements", summary="站点公告与更新动态（实时读取 config/announcements.json）")
    def announcements(response: Response) -> Dict[str, Any]:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return {"items": _load_announcements()}

    @router.get("/stats", summary="语料库公开统计信息（实时读取数据库，禁止缓存）")
    def stats(
        request: Request,
        response: Response,
        track: bool = Query(default=False, description="为 true 时记录一次页面访问（UV/PV）"),
    ) -> Dict[str, Any]:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        raw = db.get_stats()
        try:
            if track:
                visitors = metrics_db.record_visit(
                    _extract_client_ip(request),
                    request.headers.get("user-agent", ""),
                )
            else:
                visitors = metrics_db.get_summary()
        except Exception:
            visitors = {"total_uv": 0, "total_pv": 0, "today_uv": 0, "today_pv": 0}
        # 公网统计不暴露服务器本地磁盘绝对路径
        return {
            "db_size_mb": raw["db_size_mb"],
            "video_parts_total": raw["video_parts_total"],
            "video_parts_ok": raw["video_parts_ok"],
            "subtitles_total": raw["subtitles_total"],
            "subtitles_visible": raw["subtitles_visible"],
            "date_min": raw["date_min"],
            "date_max": raw["date_max"],
            "latest_updated_at": raw.get("latest_updated_at"),
            "latest_video": raw.get("latest_video"),
            "announcements": _load_announcements(),
            "visitors": visitors,
            "by_up": raw["by_up"],
            "by_scene": raw["by_scene"],
        }

    @router.get("/search", summary="关键词检索 AI 字幕语料")
    def search(
        q: str = Query(default="", description="检索关键词（空格分隔多词 AND）"),
        up_name: Optional[str] = Query(
            default=None, description="直播间（所属频道）过滤，支持逗号分隔多个，如 嘉然,向晚"
        ),
        speaker: Optional[str] = Query(
            default=None,
            description="发言人过滤（由声纹识别实现，单播默认本人），支持逗号分隔多个，如 嘉然,乃琳；可与 up_name 直播间组合使用",
        ),
        date_from: Optional[str] = Query(default=None, description="起始日期 YYYY-MM-DD"),
        date_to: Optional[str] = Query(default=None, description="截止日期 YYYY-MM-DD"),
        scene_type: Optional[str] = Query(
            default=None, description="场景过滤，支持逗号分隔，如 solo,duo,group"
        ),
        bvid: Optional[str] = Query(default=None, description="限定单个 BV 号"),
        context_lines: int = Query(default=0, ge=0, le=10, description="附带前后上下文字幕条数"),
        fuzzy: bool = Query(
            default=True,
            description="开启同音/别称/Tier 3 歧义词模糊扩展（如搜 心宜 同时命中 心仪，搜 小海诺 同时命中 小海螺）",
        ),
        speaker_filter: bool = Query(
            default=False,
            description="向后兼容旧参数；推荐使用独立的 speaker 参数与 up_name 直播间组合筛选",
        ),
        sort: str = Query(default="date_desc", pattern="^(date_desc|date_asc|relevance)$"),
        limit: int = Query(default=20, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> Dict[str, Any]:
        up_list = [x.strip() for x in up_name.split(",") if x.strip()] if up_name else None
        speaker_list = [x.strip() for x in speaker.split(",") if x.strip()] if speaker else None
        scene_list = (
            [x.strip() for x in scene_type.split(",") if x.strip()] if scene_type else None
        )
        req = SearchQuery(
            q=q,
            up_names=up_list,
            speakers=speaker_list,
            date_from=date_from,
            date_to=date_to,
            scene_types=scene_list,
            bvid=bvid,
            include_hidden=False,  # 公网接口始终只返回 is_visible=1
            fuzzy=fuzzy,
            speaker_filter=speaker_filter,
            context_lines=context_lines,
            sort=sort,
            limit=limit,
            offset=offset,
        )
        return search_svc.search(req)

    @router.get("/videos", summary="浏览已收录直播回放分 P 列表")
    def list_videos(
        up_name: Optional[str] = None,
        scene_type: Optional[str] = None,
        status: Optional[str] = Query(default="ok", description="默认仅展示已入库字幕的分 P"),
        q: Optional[str] = Query(default=None, description="按 BV 号或标题关键词筛选"),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> Dict[str, Any]:
        return search_svc.list_videos(
            up_name=up_name,
            scene_type=scene_type,
            status=status or None,
            q=q,
            include_hidden=False,
            limit=limit,
            offset=offset,
        )

    @router.get("/videos/{bvid}/subtitles", summary="按时间窗口读取某场直播的连续上下文字幕")
    def video_subtitles(
        bvid: str,
        page: int = Query(default=1, ge=1),
        start_sec: Optional[float] = Query(default=None, ge=0.0),
        end_sec: Optional[float] = Query(default=None, ge=0.0),
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> Dict[str, Any]:
        return search_svc.get_subtitle_window(
            bvid=bvid,
            page=page,
            start_sec=start_sec,
            end_sec=end_sec,
            limit=limit,
        )

    return router
