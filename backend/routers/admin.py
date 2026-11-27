"""本地私有运维管理 API（登录态配置、合集抓取任务、场景修正、数据库优化）.

重要：仅在本地管理模式（ZJ_ENABLE_ADMIN=1）下挂载，云端对外服务默认完全关闭此路由。
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import qrcode
import qrcode.image.svg
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from zhijiang_corpus.bilibili_client import BilibiliClient
from zhijiang_corpus.clip_planner import plan_clip_job
from zhijiang_corpus.cloud_ops import (
    backfill_cloud_metrics,
    deploy_code_and_announcements,
    get_asr_progress,
    get_scene_edit_summary,
    load_announcements_file,
    load_cloud_config,
    save_announcements_file,
    save_cloud_config,
    setup_ssh_key,
    sync_database_to_cloud,
    sync_scene_tags_to_cloud,
    sync_speakers_to_cloud,
)
from zhijiang_corpus.config import PROJECT_ROOT, VALID_SCENE_TYPES, load_config, save_cookie_to_env
from zhijiang_corpus.db import CorpusDB
from zhijiang_corpus.ingest import ingest_bvids, ingest_from_up
from zhijiang_corpus.scene_tagger import run_scene_tagging


class ScenePatchRequest(BaseModel):
    page: Optional[int] = Field(default=None, description="为空则更新该 BV 号全部分 P")
    scene_type: Optional[str] = Field(
        default=None,
        description=f"可选值: {', '.join(VALID_SCENE_TYPES)}",
    )
    lock: bool = Field(default=True, description="是否锁定标签避免自动初筛覆盖")
    reason: str = Field(default="manual_admin", description="备注说明")
    is_visible: Optional[int] = Field(default=None, description="1=可见，0=隐藏")


class SubtitleSpeakerPatchRequest(BaseModel):
    speaker_name: str = Field(
        default="",
        description="新发言人标签（嘉然/贝拉/乃琳/向晚/心宜/思诺/其他，支持 + 组合如 嘉然+贝拉，空串表示清空/默认）",
    )


class VideoSpeakersBatchPatchRequest(BaseModel):
    page: Optional[int] = Field(default=None, description="为空则更新该 BV 号全部分 P")
    speaker_name: str = Field(
        default="",
        description="目标发言人标签（嘉然/贝拉/乃琳/向晚/心宜/思诺/其他，支持 + 组合，空串表示清空）",
    )
    from_speaker: Optional[str] = Field(
        default="__ALL__",
        description="可选过滤：__ALL__=全部行，__EMPTY__=仅未标注行，或指定要替换的原发言人标签",
    )
    start_sec: Optional[float] = Field(default=None, ge=0.0, description="可选起始秒数（含）")
    end_sec: Optional[float] = Field(default=None, ge=0.0, description="可选结束秒数（含）")


class CookieSaveRequest(BaseModel):
    cookie: str = Field(min_length=1, description="B 站 Cookie 字符串或单独的 SESSDATA 值")


class IngestStartRequest(BaseModel):
    target: str = Field(default="all", description="UP 简称（如 嘉然）、UID 或 all")
    limit: int = Field(default=5, ge=0, le=5000, description="每位 UP 抓取条数，0 表示抓取该合集全部")
    since_date: Optional[str] = Field(default=None, description="起始日期 YYYY-MM-DD")
    until_date: Optional[str] = Field(default=None, description="截止日期 YYYY-MM-DD")
    bvids: List[str] = Field(default_factory=list, description="可选：直接指定若干 BV 号抓取")
    auto_tag_scenes: bool = Field(default=True, description="抓取结束后自动执行轻量场景初筛")
    force: bool = Field(default=False, description="是否覆盖已入库分 P")


class ClipPlanRequest(BaseModel):
    subtitle_ids: List[int] = Field(default_factory=list)
    custom_segments: List[Dict[str, Any]] = Field(default_factory=list)
    pad_before_sec: float = Field(default=1.5, ge=0.0, le=30.0)
    pad_after_sec: float = Field(default=1.5, ge=0.0, le=30.0)
    merge_gap_sec: float = Field(default=3.0, ge=0.0, le=60.0)
    output_name: str = Field(default="zhijiang_compilation.mp4")


class CloudConfigRequest(BaseModel):
    cloud_host: str = Field(min_length=1, description="例如 root@47.xx.xx.xx")
    cloud_app_dir: str = Field(default="/opt/zhijiang")
    public_url: str = Field(default="https://zjdb.mprlzy.top")
    ingest_limit: int = Field(default=10, ge=1, le=500)


class SshKeySetupRequest(BaseModel):
    cloud_host: Optional[str] = None
    ssh_password: str = Field(min_length=1, description="仅本次请求内存使用，绝不落盘")


class CloudJobRequest(BaseModel):
    mode: str = Field(
        default="db_sync_only",
        pattern="^(full_sync|db_sync_only|scene_tags_only|speakers_only|code_and_announcements|backfill_metrics)$",
    )
    ssh_password: Optional[str] = Field(default=None, description="未配置免密密钥时临时传入，绝不落盘")
    ingest_limit: int = Field(default=10, ge=1, le=500)
    run_asr: bool = Field(default=False, description="full_sync 时是否自动执行 GPU 补录")
    also_publish_announcements: bool = Field(default=True, description="同步时一并发布本地公告")
    only_locked_tags: bool = Field(default=True, description="scene_tags_only 时是否仅同步人工锁定标签")


class AnnouncementItemModel(BaseModel):
    id: str = ""
    tag: str = "公告"
    level: str = "info"
    date: str = ""
    pinned: bool = False
    active: bool = True
    content: str = ""


class AnnouncementsSaveRequest(BaseModel):
    items: List[AnnouncementItemModel] = Field(default_factory=list)


class AsrStartRequest(BaseModel):
    mode: str = Field(default="new_only", pattern="^(new_only|retranscribe_legacy)$")
    processes: int = Field(default=2, ge=1, le=4)
    batch_size: int = Field(default=12, ge=1, le=32)
    beam_size: int = Field(default=1, ge=1, le=5)
    limit: int = Field(default=0, ge=0, le=5000)


class VoiceprintStartRequest(BaseModel):
    scope: str = Field(default="multi_and_watch", pattern="^(multi_and_watch|multi_only|all)$")
    device: str = Field(default="auto", pattern="^(auto|cuda|cpu)$")
    batch_size: int = Field(default=64, ge=8, le=256)
    processes: int = Field(default=1, ge=1, le=4)
    threads: int = Field(default=4, ge=1, le=8)
    limit: int = Field(default=0, ge=0, le=5000)


class ArchiveSyncStartRequest(BaseModel):
    mode: str = Field(
        default="erchuang_all",
        pattern="^(erchuang_all|erchuang_tags_only|feishu_early|all|dry_run)$",
        description="erchuang_all=二创计画补漏+校准标签, erchuang_tags_only=仅校准标签, feishu_early=飞书早期补录, all=全部, dry_run=仅预览",
    )
    refresh_erchuang: bool = Field(default=True, description="是否实时从 B 站刷新二创计画 10 个合集最新清单")
    cutoff_date: str = Field(default="2021-08-19", description="飞书早期录播截止日期 YYYY-MM-DD")
    delay: float = Field(default=0.55, ge=0.2, le=5.0, description="B 站请求间隔秒数")


def _normalize_cookie_input(raw: str) -> str:
    text = (raw or "").strip()
    if "=" not in text and text:
        return f"SESSDATA={text}"
    return text


def _make_qr_svg_data_uri(url: str) -> str:
    qr = qrcode.QRCode(box_size=8, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(image_factory=qrcode.image.svg.SvgPathImage)
    buf = io.BytesIO()
    img.save(buf)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def build_admin_router(db: CorpusDB) -> APIRouter:
    router = APIRouter(prefix="/api/v1/admin", tags=["Local Admin API (Private)"])

    ingest_lock = threading.Lock()
    ingest_state: Dict[str, Any] = {
        "running": False,
        "current_target": "",
        "logs": [],
        "last_summary": "",
        "last_finished_at": None,
    }

    def _append_log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        with ingest_lock:
            ingest_state["logs"].append(f"[{ts}] {msg}")
            if len(ingest_state["logs"]) > 500:
                ingest_state["logs"] = ingest_state["logs"][-500:]

    @router.get("/auth/status", summary="检查本地 B 站 Cookie 登录态")
    def auth_status() -> Dict[str, Any]:
        cfg = load_config(db_path=db.db_path)
        if not cfg.bilibili_cookie:
            return {"is_login": False, "uname": "", "mid": 0, "has_sessdata": False}
        try:
            with BilibiliClient(cookie=cfg.bilibili_cookie, delay=0.0, jitter=0.0) as bili:
                return bili.verify_login()
        except Exception as exc:
            return {
                "is_login": False,
                "uname": "",
                "mid": 0,
                "has_sessdata": bool("SESSDATA=" in cfg.bilibili_cookie),
                "error": str(exc),
            }

    @router.post("/auth/cookie", summary="保存手动粘贴的 B 站 Cookie 至 .env")
    def auth_save_cookie(body: CookieSaveRequest) -> Dict[str, Any]:
        cookie_str = _normalize_cookie_input(body.cookie)
        save_cookie_to_env(cookie_str)
        with BilibiliClient(cookie=cookie_str, delay=0.0, jitter=0.0) as bili:
            info = bili.verify_login()
        return {"saved": True, **info}

    @router.post("/auth/qrcode", summary="生成 B 站官方扫码登录二维码")
    def auth_generate_qrcode() -> Dict[str, Any]:
        try:
            qr = BilibiliClient.generate_login_qrcode()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"生成 B 站登录二维码失败: {exc}")
        return {
            "qrcode_key": qr["qrcode_key"],
            "url": qr["url"],
            "qr_svg_data_uri": _make_qr_svg_data_uri(qr["url"]),
        }

    @router.get("/auth/qrcode/poll", summary="轮询 B 站扫码登录状态")
    def auth_poll_qrcode(qrcode_key: str = Query(..., min_length=4)) -> Dict[str, Any]:
        res = BilibiliClient.poll_login_qrcode(qrcode_key)
        code = int(res.get("code", -1))
        status_map = {
            0: "登录成功！Cookie 已自动保存到本地 .env",
            86101: "等待手机 B 站 App 扫码...",
            86090: "已扫码，请在手机上点击确认登录...",
            86038: "二维码已失效，请重新生成",
        }
        if code == 0 and res.get("cookie"):
            save_cookie_to_env(res["cookie"])
        return {
            "code": code,
            "message": res.get("message", ""),
            "status_text": status_map.get(code, res.get("message", "")),
        }

    @router.post("/ingest/start", summary="启动后台增量抓取任务")
    def start_ingest(body: IngestStartRequest) -> Dict[str, Any]:
        with ingest_lock:
            if ingest_state["running"]:
                raise HTTPException(status_code=409, detail="当前已有抓取任务正在运行")
            ingest_state["running"] = True
            ingest_state["logs"] = []
            ingest_state["current_target"] = body.target
            ingest_state["last_summary"] = ""

        def _worker() -> None:
            try:
                cfg = load_config(db_path=db.db_path)
                total_ok = 0
                total_lines = 0
                total_skip = 0
                total_pending_login = 0
                total_fail = 0

                if body.bvids:
                    _append_log(f"开始按指定 BV 号抓取 ({len(body.bvids)} 个)...")
                    s = ingest_bvids(
                        db,
                        cfg,
                        body.bvids,
                        force=body.force,
                        progress_cb=_append_log,
                    )
                    total_ok += s.parts_ok
                    total_lines += s.lines_inserted
                    total_skip += s.parts_skipped
                    total_pending_login += s.parts_pending_login
                    total_fail += s.parts_failed
                else:
                    targets = (
                        [m.up_name for m in cfg.members]
                        if body.target.lower() == "all"
                        else [body.target]
                    )
                    eff_limit = None if body.limit == 0 else body.limit
                    for t in targets:
                        with ingest_lock:
                            ingest_state["current_target"] = t
                        s = ingest_from_up(
                            db,
                            cfg,
                            t,
                            limit=eff_limit,
                            source="auto",
                            since_date=body.since_date,
                            until_date=body.until_date,
                            force=body.force,
                            progress_cb=_append_log,
                        )
                        total_ok += s.parts_ok
                        total_lines += s.lines_inserted
                        total_skip += s.parts_skipped
                        total_pending_login += s.parts_pending_login
                        total_fail += s.parts_failed

                if body.auto_tag_scenes and total_ok > 0:
                    _append_log("正在执行轻量场景初筛...")
                    tagged = run_scene_tagging(db, cfg, only_unknown=True, dry_run=False)
                    _append_log(f"场景初筛完成，更新 {len(tagged)} 个分 P。")

                summary_str = (
                    f"成功分P {total_ok}，新增字幕 {total_lines} 条，"
                    f"跳过 {total_skip}，待登录 {total_pending_login}，失败 {total_fail}"
                )
                _append_log(f"任务结束：{summary_str}")
                with ingest_lock:
                    ingest_state["last_summary"] = summary_str
            except Exception as exc:
                _append_log(f"抓取任务异常中止：{exc}")
                with ingest_lock:
                    ingest_state["last_summary"] = f"异常: {exc}"
            finally:
                with ingest_lock:
                    ingest_state["running"] = False
                    ingest_state["current_target"] = ""
                    ingest_state["last_finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        threading.Thread(target=_worker, daemon=True).start()
        return {"started": True, "target": body.target, "limit": body.limit}

    @router.get("/ingest/status", summary="查询后台抓取实时进度与日志")
    def ingest_status() -> Dict[str, Any]:
        with ingest_lock:
            return dict(ingest_state)

    @router.post("/scenes/tag", summary="触发轻量规则场景初筛")
    def api_tag_scenes(only_unknown: bool = False) -> Dict[str, Any]:
        cfg = load_config(db_path=db.db_path)
        res = run_scene_tagging(db, cfg, only_unknown=only_unknown, dry_run=False)
        skipped = sum(1 for r in res if r.skipped_locked)
        return {"processed": len(res), "skipped_locked": skipped}

    @router.patch("/videos/{bvid}/scene", summary="人工修正并锁定场景标签或可见性")
    def patch_video_scene(bvid: str, body: ScenePatchRequest) -> Dict[str, Any]:
        updated = 0
        if body.scene_type is not None:
            if body.scene_type not in VALID_SCENE_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"无效 scene_type，可选值: {VALID_SCENE_TYPES}",
                )
            updated += db.update_scene_type(
                bvid,
                body.page,
                body.scene_type,
                confidence=1.0,
                reason=body.reason,
                lock=body.lock,
            )
        if body.is_visible is not None:
            updated += db.update_visibility(bvid, body.page, body.is_visible)
        if updated == 0:
            raise HTTPException(status_code=404, detail=f"未找到视频记录: {bvid}")
        return {"status": "ok", "bvid": bvid, "page": body.page, "affected_rows": updated}

    @router.patch("/subtitles/{subtitle_id}/speaker", summary="人工修正单条字幕的发言人标签")
    def patch_subtitle_speaker(subtitle_id: int, body: SubtitleSpeakerPatchRequest) -> Dict[str, Any]:
        try:
            res = db.update_subtitle_speaker(subtitle_id, body.speaker_name, confidence=1.0)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if res is None:
            raise HTTPException(status_code=404, detail=f"未找到字幕记录: {subtitle_id}")
        return {"status": "ok", **res}

    @router.patch("/videos/{bvid}/speakers", summary="按场次/分P/时间段批量修正发言人标签")
    def patch_video_speakers(bvid: str, body: VideoSpeakersBatchPatchRequest) -> Dict[str, Any]:
        if (
            body.start_sec is not None
            and body.end_sec is not None
            and body.start_sec > body.end_sec
        ):
            raise HTTPException(status_code=400, detail="起始秒数不能大于结束秒数")
        try:
            res = db.batch_update_video_speakers(
                bvid,
                body.page,
                body.speaker_name,
                from_speaker=body.from_speaker,
                start_sec=body.start_sec,
                end_sec=body.end_sec,
                confidence=1.0,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not res.get("found"):
            raise HTTPException(status_code=404, detail=f"未找到视频记录: {bvid}")
        return {"status": "ok", **res}

    @router.post("/db/optimize", summary="大批量抓取后优化 FTS5 索引与回收 WAL")
    def optimize_db() -> Dict[str, Any]:
        return db.optimize_db()

    @router.post("/clips/plan", summary="预留：二阶段 FFmpeg 切片清单规划")
    def create_clip_plan(body: ClipPlanRequest) -> Dict[str, Any]:
        if not body.subtitle_ids and not body.custom_segments:
            raise HTTPException(status_code=400, detail="请至少提供 subtitle_ids 或 custom_segments")
        return plan_clip_job(
            db,
            subtitle_ids=body.subtitle_ids,
            custom_segments=body.custom_segments,
            pad_before_sec=body.pad_before_sec,
            pad_after_sec=body.pad_after_sec,
            merge_gap_sec=body.merge_gap_sec,
            output_name=body.output_name,
        )

    # ── 云端运维、公告管理、纯标签同步与本地 GPU ASR 控制中心 ──
    cloud_lock = threading.Lock()
    cloud_state: Dict[str, Any] = {
        "running": False,
        "mode": "",
        "logs": [],
        "last_summary": "",
        "last_error": "",
        "last_finished_at": None,
        "live_stats": None,
    }

    def _cloud_log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        with cloud_lock:
            cloud_state["logs"].append(f"[{ts}] {msg}")
            if len(cloud_state["logs"]) > 500:
                cloud_state["logs"] = cloud_state["logs"][-500:]

    @router.get("/cloud/config", summary="读取本地云端同步配置与免密密钥状态")
    def api_get_cloud_config() -> Dict[str, Any]:
        return load_cloud_config()

    @router.post("/cloud/config", summary="保存云端服务器地址配置至本地 .cloud_sync.env（不保存密码）")
    def api_save_cloud_config(body: CloudConfigRequest) -> Dict[str, Any]:
        return save_cloud_config(
            cloud_host=body.cloud_host,
            cloud_app_dir=body.cloud_app_dir,
            public_url=body.public_url,
            ingest_limit=body.ingest_limit,
        )

    @router.post("/cloud/setup-ssh-key", summary="一键配置本机 SSH 免密密钥到云端（密码仅内存使用一次）")
    def api_setup_ssh_key(body: SshKeySetupRequest) -> Dict[str, Any]:
        conf = load_cloud_config()
        target_host = (body.cloud_host or conf.get("cloud_host") or "").strip()
        if not target_host:
            raise HTTPException(status_code=400, detail="请先填写云端服务器地址")
        logs: List[str] = []
        try:
            res = setup_ssh_key(target_host, body.ssh_password, logs.append)
            return {**res, "logs": logs, "config": load_cloud_config()}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/cloud/run", summary="在后台执行云端同步 / 纯标签同步 / 仅发布代码与公告")
    def api_run_cloud_job(body: CloudJobRequest) -> Dict[str, Any]:
        conf = load_cloud_config()
        if not conf.get("cloud_host"):
            raise HTTPException(status_code=400, detail="请先在上方配置并保存云端服务器地址")
        with cloud_lock:
            if cloud_state["running"]:
                raise HTTPException(status_code=409, detail="当前已有云端同步任务正在运行，请稍候")
            cloud_state["running"] = True
            cloud_state["mode"] = body.mode
            cloud_state["logs"] = []
            cloud_state["last_summary"] = ""
            cloud_state["last_error"] = ""

        # 密码仅留在局部变量供本线程一次使用，完成后立刻丢弃
        mem_pwd = body.ssh_password

        def _worker() -> None:
            try:
                if body.mode == "code_and_announcements":
                    res = deploy_code_and_announcements(conf, mem_pwd, _cloud_log)
                    summary = "线上代码与公告发布完成"
                elif body.mode == "backfill_metrics":
                    res = backfill_cloud_metrics(conf, mem_pwd, _cloud_log)
                    v = (res.get("backfill") or {}).get("visitors") or {}
                    summary = f"历史访客补回完成：累计 {v.get('total_uv', 0)} 人 / {v.get('total_pv', 0)} 次"
                elif body.mode == "speakers_only":
                    res = sync_speakers_to_cloud(
                        db,
                        conf,
                        mem_pwd,
                        also_publish_announcements=body.also_publish_announcements,
                        log_cb=_cloud_log,
                    )
                    ap = res.get("applied") or {}
                    summary = (
                        f"声纹标注同步完成：更新 {ap.get('parts_applied', 0)} 个分P / "
                        f"{ap.get('lines_updated', 0):,} 条发言人标签"
                    )
                elif body.mode == "scene_tags_only":
                    res = sync_scene_tags_to_cloud(
                        db,
                        conf,
                        mem_pwd,
                        only_locked=body.only_locked_tags,
                        also_publish_announcements=body.also_publish_announcements,
                        log_cb=_cloud_log,
                    )
                    ap = res.get("applied") or {}
                    summary = (
                        f"纯标签同步完成：更新 {ap.get('parts_updated', 0)} 个分P / "
                        f"{ap.get('bvids_updated', 0)} 场直播"
                    )
                else:
                    res = sync_database_to_cloud(
                        db,
                        conf,
                        mem_pwd,
                        run_ingest=(body.mode == "full_sync"),
                        run_asr=(body.mode == "full_sync" and body.run_asr),
                        ingest_limit=body.ingest_limit,
                        also_publish_announcements=body.also_publish_announcements,
                        log_cb=_cloud_log,
                    )
                    ap = res.get("applied") or {}
                    summary = (
                        f"数据库增量同步完成：写入 {ap.get('parts_applied', 0)} 个分P / "
                        f"{ap.get('subtitles_inserted', 0):,} 条字幕"
                    )
                with cloud_lock:
                    cloud_state["last_summary"] = summary
                    cloud_state["live_stats"] = res.get("live_stats")
            except Exception as exc:
                _cloud_log(f"❌ 任务失败：{exc}")
                with cloud_lock:
                    cloud_state["last_error"] = str(exc)
                    cloud_state["last_summary"] = f"失败：{exc}"
            finally:
                with cloud_lock:
                    cloud_state["running"] = False
                    cloud_state["last_finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        threading.Thread(target=_worker, daemon=True).start()
        return {"started": True, "mode": body.mode}

    @router.get("/cloud/status", summary="查询云端运维任务实时日志与状态")
    def api_cloud_status() -> Dict[str, Any]:
        with cloud_lock:
            return dict(cloud_state)

    @router.get("/announcements", summary="读取本地 config/announcements.json 全部公告（含未生效）")
    def api_get_announcements() -> Dict[str, Any]:
        return {"items": load_announcements_file()}

    @router.put("/announcements", summary="保存公告列表至本地 config/announcements.json")
    def api_save_announcements(body: AnnouncementsSaveRequest) -> Dict[str, Any]:
        saved = save_announcements_file([it.model_dump() for it in body.items])
        return {"saved": True, "items": saved}

    @router.get("/scenes/summary", summary="统计人工锁定场景标签数量（今日与累计）")
    def api_scenes_summary() -> Dict[str, Any]:
        return get_scene_edit_summary(db)

    # ── 本地 GPU ASR 补录/重录托管控制 ──
    asr_lock = threading.Lock()
    asr_procs: List[subprocess.Popen] = []
    asr_state: Dict[str, Any] = {
        "running": False,
        "mode": "",
        "logs": [],
        "last_finished_at": None,
    }

    def _asr_log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        with asr_lock:
            asr_state["logs"].append(f"[{ts}] {msg}")
            if len(asr_state["logs"]) > 400:
                asr_state["logs"] = asr_state["logs"][-400:]

    def _count_external_python_procs(needle: str) -> int:
        if sys.platform != "win32":
            return 0
        try:
            out = subprocess.check_output(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"(Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object {{ $_.CommandLine -like '*{needle}*' }} | Measure-Object).Count",
                ],
                text=True,
                timeout=3,
                stderr=subprocess.DEVNULL,
            ).strip()
            return int(out) if out.isdigit() else 0
        except Exception:
            return 0

    def _recent_db_activity_logs(kind: str, limit: int = 15) -> List[str]:
        with db.session() as conn:
            if kind == "vp":
                rows = conn.execute(
                    """
                    SELECT updated_at, up_name, bvid, page, scene_type, line_count, title
                      FROM videos
                     WHERE voiceprint_status = 'ok'
                     ORDER BY updated_at DESC
                     LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [
                    f"[{str(r['updated_at'])[11:19]} UTC] ✅ 完成声纹 {r['up_name']} {r['bvid']} P{r['page']} "
                    f"({r['scene_type']}, {r['line_count']}行) | {str(r['title'] or '')[:38]}"
                    for r in reversed(rows)
                ]
            rows = conn.execute(
                """
                SELECT updated_at, up_name, bvid, page, subtitle_lan, line_count, title
                  FROM videos
                 WHERE subtitle_lan LIKE 'asr-%' AND subtitle_status = 'ok'
                 ORDER BY updated_at DESC
                 LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [
                f"[{str(r['updated_at'])[11:19]} UTC] ✅ 完成ASR {r['up_name']} {r['bvid']} P{r['page']} "
                f"({r['subtitle_lan']}, {r['line_count']}行) | {str(r['title'] or '')[:38]}"
                for r in reversed(rows)
            ]

    @router.get("/asr/status", summary="查看本地 GPU ASR 补录/重录实时进度与托管状态")
    def api_asr_status() -> Dict[str, Any]:
        progress = get_asr_progress(db)
        with asr_lock:
            alive = [p for p in asr_procs if p.poll() is None]
            asr_state["running"] = len(alive) > 0
            state_copy = dict(asr_state)
        if not state_copy["running"]:
            ext_n = _count_external_python_procs("backfill-asr")
            if ext_n > 0:
                state_copy["running"] = True
                state_copy["external"] = True
                state_copy["mode"] = f"外部运行中 ({ext_n} 进程)"
                if not state_copy["logs"]:
                    state_copy["logs"] = [
                        f"[外部托管] 检测到 {ext_n} 个 GPU ASR 进程正在后台运行，以下为最近完成入库的分P（每 5 秒自动刷新）：",
                        *_recent_db_activity_logs("asr", limit=15),
                    ]
        return {"progress": progress, "job": state_copy}

    @router.post("/asr/start", summary="从本地后台启动 GPU 语音补录或旧 ASR 高精度重录")
    def api_asr_start(body: AsrStartRequest) -> Dict[str, Any]:
        with asr_lock:
            alive = [p for p in asr_procs if p.poll() is None]
            if alive:
                raise HTTPException(status_code=409, detail="当前已有本地后台托管的 ASR 任务正在运行")
            asr_procs.clear()
            asr_state["running"] = True
            asr_state["mode"] = body.mode
            asr_state["logs"] = []

        env = os.environ.copy()
        hf_dir = PROJECT_ROOT / "data" / "cache" / "huggingface"
        tmp_dir = PROJECT_ROOT / "data" / "tmp_asr"
        hf_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        env["HF_HOME"] = str(hf_dir)
        env["TMP"] = str(tmp_dir)
        env["TEMP"] = str(tmp_dir)
        env["PYTHONIOENCODING"] = "utf-8"

        py = sys.executable
        n_procs = max(1, min(4, int(body.processes)))
        beam_sz = max(1, min(5, int(body.beam_size)))

        def _spawn_and_watch() -> None:
            try:
                _asr_log(
                    f"启动 GPU ASR 任务：mode={body.mode}，进程数={n_procs}，"
                    f"batch_size={body.batch_size}，beam_size={beam_sz}"
                )
                local_procs: List[subprocess.Popen] = []
                for idx in range(n_procs):
                    cmd = [
                        py,
                        "-u",
                        "run.py",
                        "backfill-asr",
                        "--limit",
                        str(body.limit),
                        "--batch-size",
                        str(body.batch_size),
                        "--beam-size",
                        str(beam_sz),
                        "--chunk-length",
                        "10",
                        "--workers",
                        "1",
                        "--prefetch",
                        "1",
                    ]
                    if body.mode == "retranscribe_legacy":
                        cmd.append("--retranscribe-legacy-asr")
                    if n_procs > 1:
                        cmd.extend(["--shard-index", str(idx), "--shard-total", str(n_procs)])
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(PROJECT_ROOT),
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    local_procs.append(proc)
                    with asr_lock:
                        asr_procs.append(proc)

                def _pump(p: subprocess.Popen, shard: int) -> None:
                    if not p.stdout:
                        return
                    for line in p.stdout:
                        txt = line.rstrip()
                        if txt:
                            prefix = f"[GPU#{shard + 1}] " if n_procs > 1 else ""
                            _asr_log(f"{prefix}{txt}")

                threads = [
                    threading.Thread(target=_pump, args=(p, i), daemon=True)
                    for i, p in enumerate(local_procs)
                ]
                for t in threads:
                    t.start()
                for p in local_procs:
                    p.wait()
                for t in threads:
                    t.join(timeout=2)
                _asr_log("GPU ASR 任务已结束。")
            except Exception as exc:
                _asr_log(f"❌ GPU ASR 任务异常：{exc}")
            finally:
                # 清理已结束/被强杀的 ASR 临时音频目录，防止残留占用磁盘
                tmp_base = PROJECT_ROOT / "data" / "tmp_asr"
                for d in tmp_base.glob("zj_asr_*"):
                    shutil.rmtree(d, ignore_errors=True)
                with asr_lock:
                    asr_procs.clear()
                    asr_state["running"] = False
                    asr_state["last_finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        threading.Thread(target=_spawn_and_watch, daemon=True).start()
        return {"started": True, "mode": body.mode, "processes": n_procs}

    @router.post("/asr/stop", summary="安全停止由本地后台启动的 GPU ASR 子进程")
    def api_asr_stop() -> Dict[str, Any]:
        stopped = 0
        with asr_lock:
            for p in asr_procs:
                if p.poll() is None:
                    if sys.platform == "win32":
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(p.pid)],
                            capture_output=True,
                            check=False,
                        )
                    else:
                        p.terminate()
                    stopped += 1
        if stopped:
            _asr_log(f"已发送停止信号给 {stopped} 个 GPU ASR 进程树（已写入数据库的分P全部保留）。")
        return {"stopped": stopped}

    # ── 本地声纹发言人标注控制台（支持断点续跑与多进程并行）──
    vp_lock = threading.Lock()
    vp_procs: List[subprocess.Popen] = []
    vp_state: Dict[str, Any] = {
        "running": False,
        "scope": "",
        "logs": [],
        "last_finished_at": None,
    }

    def _vp_log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        with vp_lock:
            vp_state["logs"].append(f"[{ts}] {msg}")
            if len(vp_state["logs"]) > 400:
                vp_state["logs"] = vp_state["logs"][-400:]

    @router.get("/voiceprint/status", summary="查看声纹发言人标注实时进度与日志")
    def api_voiceprint_status() -> Dict[str, Any]:
        from zhijiang_corpus.voiceprint import get_voiceprint_progress

        progress = get_voiceprint_progress(db)
        with vp_lock:
            alive = [p for p in vp_procs if p.poll() is None]
            vp_state["running"] = len(alive) > 0
            st_copy = dict(vp_state)
        if not st_copy["running"]:
            ext_n = _count_external_python_procs("run-voiceprint")
            if ext_n > 0:
                st_copy["running"] = True
                st_copy["external"] = True
                st_copy["scope"] = f"外部运行中 ({ext_n} 进程)"
                if not st_copy["logs"]:
                    st_copy["logs"] = [
                        f"[外部托管] 检测到 {ext_n} 个声纹标注进程正在后台运行，以下为最近完成标注的分P（每 5 秒自动刷新）：",
                        *_recent_db_activity_logs("vp", limit=15),
                    ]
        return {"progress": progress, "job": st_copy}

    @router.post("/voiceprint/start", summary="启动本地批量声纹发言人标注（支持断点续跑）")
    def api_voiceprint_start(body: VoiceprintStartRequest) -> Dict[str, Any]:
        with vp_lock:
            alive = [p for p in vp_procs if p.poll() is None]
            if alive:
                raise HTTPException(status_code=409, detail="当前已有声纹标注任务正在运行")
            vp_procs.clear()
            vp_state["running"] = True
            vp_state["scope"] = body.scope
            vp_state["logs"] = []

        env = os.environ.copy()
        tmp_dir = PROJECT_ROOT / "data" / "tmp_asr"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        env["TMP"] = str(tmp_dir)
        env["TEMP"] = str(tmp_dir)
        env["PYTHONIOENCODING"] = "utf-8"
        env["CUDA_CACHE_MAXSIZE"] = "2147483648"

        py = sys.executable
        n_procs = max(1, min(4, int(body.processes)))
        prefetch_per_proc = 3 if n_procs == 1 else 2

        def _spawn_vp() -> None:
            try:
                _vp_log(
                    f"启动声纹发言人标注：scope={body.scope}，device={body.device}，"
                    f"batch_size={body.batch_size}，并行进程={n_procs}，每进程并发拉流={prefetch_per_proc}"
                )
                local_procs: List[subprocess.Popen] = []
                for idx in range(n_procs):
                    cmd = [
                        py,
                        "-u",
                        "run.py",
                        "run-voiceprint",
                        "--scope",
                        body.scope,
                        "--device",
                        body.device,
                        "--batch-size",
                        str(body.batch_size),
                        "--limit",
                        str(body.limit),
                        "--threads",
                        str(body.threads),
                        "--prefetch",
                        str(prefetch_per_proc),
                    ]
                    if n_procs > 1:
                        cmd.extend(["--shard-index", str(idx), "--shard-total", str(n_procs)])
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(PROJECT_ROOT),
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    local_procs.append(proc)
                    with vp_lock:
                        vp_procs.append(proc)

                def _pump(p: subprocess.Popen, shard: int) -> None:
                    if not p.stdout:
                        return
                    for line in p.stdout:
                        txt = line.rstrip()
                        if txt:
                            prefix = f"[VP#{shard + 1}] " if n_procs > 1 else ""
                            _vp_log(f"{prefix}{txt}")

                threads_p = [
                    threading.Thread(target=_pump, args=(p, i), daemon=True)
                    for i, p in enumerate(local_procs)
                ]
                for t in threads_p:
                    t.start()
                for p in local_procs:
                    p.wait()
                for t in threads_p:
                    t.join(timeout=2)
                _vp_log("声纹发言人标注任务已结束。")
            except Exception as exc:
                _vp_log(f"❌ 声纹任务异常：{exc}")
            finally:
                # 清理已结束/被强杀的声纹预取临时音频，防止残留堆积数 GB WAV
                vp_tmp = PROJECT_ROOT / "data" / "tmp_asr" / "zhijiang_vp"
                if vp_tmp.exists():
                    for f in list(vp_tmp.glob("*")):
                        try:
                            if f.is_dir():
                                shutil.rmtree(f, ignore_errors=True)
                            else:
                                f.unlink(missing_ok=True)
                        except OSError:
                            pass
                with vp_lock:
                    vp_procs.clear()
                    vp_state["running"] = False
                    vp_state["last_finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        threading.Thread(target=_spawn_vp, daemon=True).start()
        return {"started": True, "scope": body.scope, "processes": n_procs}

    @router.post("/voiceprint/stop", summary="安全停止声纹发言人标注进程树")
    def api_voiceprint_stop() -> Dict[str, Any]:
        stopped = 0
        with vp_lock:
            for p in vp_procs:
                if p.poll() is None:
                    if sys.platform == "win32":
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(p.pid)],
                            capture_output=True,
                            check=False,
                        )
                    else:
                        p.terminate()
                    stopped += 1
        if stopped:
            _vp_log(f"已安全停止 {stopped} 个声纹标注进程（已写库的分P全部保留，随时可续跑）。")
        return {"stopped": stopped}

    # ── 第三方归档补录（飞书早期录播 + 二创计画 10 合集补漏与标签校准）──
    archive_lock = threading.Lock()
    archive_proc_box: List[subprocess.Popen] = []
    archive_state: Dict[str, Any] = {
        "running": False,
        "mode": "",
        "logs": [],
        "last_summary": "",
        "last_finished_at": None,
    }

    def _archive_log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        with archive_lock:
            archive_state["logs"].append(f"[{ts}] {msg}")
            if len(archive_state["logs"]) > 500:
                archive_state["logs"] = archive_state["logs"][-500:]

    def _get_archive_db_summary() -> Dict[str, Any]:
        with db.session() as conn:
            rows = conn.execute(
                """
                SELECT COALESCE(json_extract(extra_meta, '$.archive_source'), 'official') AS src,
                       COUNT(DISTINCT bvid) AS videos,
                       COUNT(*) AS parts,
                       SUM(CASE WHEN subtitle_status = 'ok' THEN 1 ELSE 0 END) AS ok_parts,
                       SUM(CASE WHEN subtitle_status = 'no_subtitle' THEN 1 ELSE 0 END) AS no_sub_parts,
                       COALESCE(SUM(line_count), 0) AS lines
                  FROM videos
                 GROUP BY src
                """
            ).fetchall()
        by_src = {str(r["src"]): dict(r) for r in rows}
        return {
            "feishu_early": by_src.get(
                "feishu_early",
                {"videos": 0, "parts": 0, "ok_parts": 0, "no_sub_parts": 0, "lines": 0},
            ),
            "erchuang_missing": by_src.get(
                "erchuang_missing",
                {"videos": 0, "parts": 0, "ok_parts": 0, "no_sub_parts": 0, "lines": 0},
            ),
            "official": by_src.get(
                "official",
                {"videos": 0, "parts": 0, "ok_parts": 0, "no_sub_parts": 0, "lines": 0},
            ),
        }

    @router.get("/archives/status", summary="查看飞书早期录播与二创计画补录/标签校准状态与日志")
    def api_archives_status() -> Dict[str, Any]:
        summary = _get_archive_db_summary()
        with archive_lock:
            alive = [p for p in archive_proc_box if p.poll() is None]
            archive_state["running"] = len(alive) > 0
            job_copy = dict(archive_state)
        return {"summary": summary, "job": job_copy}

    @router.post("/archives/start", summary="在后台执行飞书早期录播补录或二创计画合集补漏与标签校准")
    def api_archives_start(body: ArchiveSyncStartRequest) -> Dict[str, Any]:
        with archive_lock:
            alive = [p for p in archive_proc_box if p.poll() is None]
            if alive:
                raise HTTPException(status_code=409, detail="当前已有归档补录/校准任务正在运行，请稍候")
            archive_proc_box.clear()
            archive_state["running"] = True
            archive_state["mode"] = body.mode
            archive_state["logs"] = []
            archive_state["last_summary"] = ""

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        cmd = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "scripts" / "sync_archives.py"),
            "--db",
            str(db.db_path),
            "--cutoff-date",
            body.cutoff_date,
            "--delay",
            str(body.delay),
        ]
        if body.refresh_erchuang and body.mode != "feishu_early":
            cmd.append("--refresh-erchuang")
        if body.mode == "dry_run":
            cmd.append("--dry-run")
        elif body.mode == "erchuang_all":
            cmd.append("--only-erchuang")
        elif body.mode == "erchuang_tags_only":
            cmd.append("--only-tags")
        elif body.mode == "feishu_early":
            cmd.append("--only-feishu")

        def _spawn_archive() -> None:
            try:
                _archive_log(
                    f"启动归档同步任务：mode={body.mode}，refresh_erchuang={body.refresh_erchuang}，"
                    f"cutoff_date={body.cutoff_date}"
                )
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                with archive_lock:
                    archive_proc_box.append(proc)

                last_important = ""
                if proc.stdout:
                    for line in proc.stdout:
                        txt = line.rstrip()
                        if txt:
                            _archive_log(txt)
                            if any(
                                k in txt
                                for k in ("补录完成", "标签修正完成", "优化完成", "二创计画对齐规划")
                            ):
                                last_important = txt
                rc = proc.wait()
                summary = (
                    last_important or f"任务完成 (exit={rc})"
                    if rc == 0
                    else f"任务异常退出 (exit={rc})"
                )
                _archive_log(f"归档同步任务结束：{summary}")
                with archive_lock:
                    archive_state["last_summary"] = summary
            except Exception as exc:
                _archive_log(f"❌ 归档同步任务异常：{exc}")
                with archive_lock:
                    archive_state["last_summary"] = f"异常：{exc}"
            finally:
                with archive_lock:
                    archive_proc_box.clear()
                    archive_state["running"] = False
                    archive_state["last_finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        threading.Thread(target=_spawn_archive, daemon=True).start()
        return {"started": True, "mode": body.mode}

    @router.post("/archives/stop", summary="安全停止运行中的归档补录/标签校准子进程")
    def api_archives_stop() -> Dict[str, Any]:
        stopped = 0
        with archive_lock:
            for p in archive_proc_box:
                if p.poll() is None:
                    if sys.platform == "win32":
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(p.pid)],
                            capture_output=True,
                            check=False,
                        )
                    else:
                        p.terminate()
                    stopped += 1
        if stopped:
            _archive_log(f"已安全停止归档同步任务（已入库场次与已修改标签全部保留，随时可续跑）。")
        return {"stopped": stopped}

    return router
