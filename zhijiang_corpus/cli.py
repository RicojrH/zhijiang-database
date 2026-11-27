"""统一命令行入口：建库、B站登录校验、合集抓取入库、场景初筛、搜索查询与本地检索软件."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import qrcode
import uvicorn

from .bilibili_client import BilibiliClient
from .clip_planner import plan_clip_job
from .config import VALID_SCENE_TYPES, load_config, save_cookie_to_env
from .db import CorpusDB
from .ingest import ingest_bvids, ingest_from_up, ingest_local_file, seed_demo_data
from .metrics import SiteMetricsDB
from .normalizer import run_corpus_normalization
from .scene_tagger import run_scene_tagging
from .search import SearchQuery, SearchService
from .sync_delta import (
    apply_delta,
    apply_scene_tags,
    apply_speakers,
    build_manifest,
    export_delta,
    export_scene_tags,
    export_speakers,
)


def _print_json(data: object) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def cmd_init_db(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db)
    db = CorpusDB(cfg.db_path)
    db.init_db()
    print(f"[OK] 数据库初始化完成: {db.db_path}")
    return 0


def cmd_check_auth(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db, cookie=args.cookie)
    if not cfg.bilibili_cookie:
        print("[WARN] 当前未配置 BILIBILI_COOKIE / SESSDATA（可用 `python run.py login` 扫码登录）")
        return 1
    with BilibiliClient(cookie=cfg.bilibili_cookie, delay=0.0, jitter=0.0) as bili:
        info = bili.verify_login()
    _print_json(info)
    return 0 if info.get("is_login") else 1


def cmd_login(args: argparse.Namespace) -> int:
    """在终端展示 B 站官方登录二维码，手机 B 站扫码后自动保存 Cookie 到 .env."""
    qr_info = BilibiliClient.generate_login_qrcode()
    url = qr_info["url"]
    key = qr_info["qrcode_key"]

    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    print("=== 请使用手机 B 站 App 扫描下方二维码登录 ===")
    qr.print_ascii(invert=True)
    print(f"(也可直接打开本地软件 http://127.0.0.1:8000 在「抓取入库与登录管理」页面扫码)")

    deadline = time.time() + 180
    last_code = None
    while time.time() < deadline:
        res = BilibiliClient.poll_login_qrcode(key)
        code = int(res.get("code", -1))
        if code != last_code:
            if code == 86101:
                print("等待扫码...")
            elif code == 86090:
                print("已扫码，请在手机上点击确认登录...")
            last_code = code
        if code == 0 and res.get("cookie"):
            env_path = save_cookie_to_env(res["cookie"])
            with BilibiliClient(cookie=res["cookie"], delay=0.0, jitter=0.0) as bili:
                info = bili.verify_login()
            print(f"[OK] 登录成功！账号：{info.get('uname')} (UID: {info.get('mid')})")
            print(f"[OK] Cookie 已自动保存至: {env_path}")
            return 0
        if code == 86038:
            print("[ERROR] 二维码已失效，请重新执行 `python run.py login`。")
            return 1
        time.sleep(2)
    print("[ERROR] 扫码超时，请重试。")
    return 1


def cmd_list_collections(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db)
    targets = [m for m in cfg.members] if args.up.lower() == "all" else [cfg.resolve_member(args.up)]
    with BilibiliClient(cookie=cfg.bilibili_cookie, delay=0.5, jitter=0.2) as bili:
        for m in targets:
            if m is None:
                continue
            cols = bili.discover_collections(m.uid)
            print(f"=== {m.up_name} ({m.official_name}, UID={m.uid}) ===")
            for c in cols:
                mark = "  <-- [已配置直播回放]" if c.collection_id == m.replay_series_id else ""
                print(f"  [{c.kind}] {c.collection_id} | {c.name} (total={c.total}){mark}")
    return 0


def cmd_seed_demo(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db)
    db = CorpusDB(cfg.db_path)
    res = seed_demo_data(db)
    print(
        f"[OK] 已写入演示语料: {res['videos']} 场视频，"
        f"新增字幕 {res['lines_inserted']} 条，忽略重复 {res['lines_duplicate']} 条"
    )
    return 0


def cmd_ingest_bvid(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db, cookie=args.cookie)
    db = CorpusDB(cfg.db_path)
    if not cfg.bilibili_cookie:
        print(
            "[WARN] 未检测到 BILIBILI_COOKIE / BILIBILI_SESSDATA。"
            "B 站 AI 字幕 (ai-zh) 需要登录 Cookie，建议先运行 `python run.py login`。"
        )
    pages = [int(p) for p in args.pages.split(",") if p.strip()] if args.pages else None
    summary = ingest_bvids(
        db,
        cfg,
        args.bvids,
        up_name_override=args.up_name,
        pages_filter=pages,
        force=args.force,
        retry_no_sub=args.retry_no_sub,
        progress_cb=lambda m: print(m, flush=True),
    )
    _print_json(summary.__dict__)
    return 0 if summary.parts_failed == 0 else 1


def cmd_ingest_up(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db, cookie=args.cookie)
    db = CorpusDB(cfg.db_path)
    if not cfg.bilibili_cookie:
        print(
            "[WARN] 未检测到 BILIBILI_COOKIE / BILIBILI_SESSDATA。"
            "B 站 AI 字幕 (ai-zh) 需要登录态，请先执行 `python run.py login` 或配置 .env。"
        )
    targets = [m.up_name for m in cfg.members] if args.up.lower() == "all" else [args.up]
    eff_limit = None if args.limit == 0 else args.limit
    overall = []
    has_err = False
    try:
        for target in targets:
            print(
                f"==> 开始拉取 UP 主 [{target}] (source={args.source}, limit={eff_limit}) ...",
                flush=True,
            )
            summary = ingest_from_up(
                db,
                cfg,
                target,
                limit=eff_limit,
                source=args.source,
                series_id=args.series_id,
                season_id=args.season_id,
                keyword=args.keyword,
                title_contains=args.title_contains,
                since_date=args.since,
                until_date=args.until,
                force=args.force,
                retry_no_sub=args.retry_no_sub,
                progress_cb=lambda m: print(m, flush=True),
            )
            overall.append({"target": target, **summary.__dict__})
            if summary.parts_failed > 0:
                has_err = True
    except KeyboardInterrupt:
        print(
            "\n[INTERRUPTED] 收到 Ctrl+C，已入库的场次均已安全提交到 SQLite。"
            "下次直接重新运行相同命令即可秒级跳过已入库视频，继续断点续抓！"
        )
    _print_json(overall)
    return 0 if not has_err else 1


def cmd_retry_failed(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db, cookie=args.cookie)
    db = CorpusDB(cfg.db_path)
    db.init_db()
    bvids = db.list_retry_bvids(include_no_subtitle=args.include_no_sub)
    if not bvids:
        print("[OK] 当前没有需要重试的失败/待处理场次。")
        return 0
    print(f"==> 共发现 {len(bvids)} 个待重试 BV 号，开始补抓...", flush=True)
    summary = ingest_bvids(
        db,
        cfg,
        bvids,
        retry_no_sub=args.include_no_sub,
        progress_cb=lambda m: print(m, flush=True),
    )
    _print_json(summary.__dict__)
    return 0 if summary.parts_failed == 0 else 1


def cmd_optimize_db(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db)
    db = CorpusDB(cfg.db_path)
    res = db.optimize_db()
    _print_json(res)
    return 0


def cmd_normalize_corpus(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db)
    db = CorpusDB(cfg.db_path)
    res = run_corpus_normalization(
        db,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        progress_cb=lambda m: print(m, flush=True),
    )
    _print_json(res)
    return 0


def cmd_backfill_asr(args: argparse.Namespace) -> int:
    from .asr_backfill import run_asr_backfill

    cfg = load_config(db_path=args.db, cookie=args.cookie)
    db = CorpusDB(cfg.db_path)
    eff_limit = None if args.limit == 0 else args.limit
    try:
        summary = run_asr_backfill(
            db,
            cfg,
            bvid=args.bvid,
            page=args.page,
            up_name=args.up,
            limit=eff_limit,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            shortest_first=args.shortest_first,
            force=args.force,
            retranscribe_legacy_asr=args.retranscribe_legacy_asr,
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            batch_size=args.batch_size,
            beam_size=args.beam_size,
            word_timestamps=not args.no_word_timestamps,
            chunk_length=args.chunk_length,
            workers=args.workers,
            prefetch=args.prefetch,
            shard_index=args.shard_index,
            shard_total=args.shard_total,
            use_hf_mirror=args.hf_mirror,
            keep_audio=args.keep_audio,
            progress_cb=lambda m: print(m, flush=True),
        )
        _print_json(summary.__dict__)
        return 0 if summary.parts_failed == 0 else 1
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] 收到 Ctrl+C，已完成转录的分 P 均已安全提交入库，下次重跑会自动跳过。")
        return 0


def cmd_resplit_asr(args: argparse.Namespace) -> int:
    from .asr_backfill import resplit_legacy_asr

    cfg = load_config(db_path=args.db)
    db = CorpusDB(cfg.db_path)
    res = resplit_legacy_asr(
        db,
        max_chars=args.max_chars,
        dry_run=args.dry_run,
        progress_cb=lambda m: print(m, flush=True),
    )
    _print_json(res)
    return 0


def cmd_ingest_local(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db)
    db = CorpusDB(cfg.db_path)
    ins, dup = ingest_local_file(
        db,
        args.file,
        up_name=cfg.resolve_up_name(raw_name=args.up_name),
        bvid=args.bvid,
        page=args.page,
        pub_date=args.pub_date,
        title=args.title or "",
        part_title=args.part_title or "",
        scene_type=args.scene_type,
        force=args.force,
    )
    print(f"[OK] 本地字幕导入完成: 新增 {ins} 条，忽略重复 {dup} 条")
    return 0


def cmd_tag_scenes(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db)
    db = CorpusDB(cfg.db_path)
    results = run_scene_tagging(
        db,
        cfg,
        bvid=args.bvid,
        only_unknown=args.only_unknown,
        fallback_solo=not args.keep_unknown_solo,
        dry_run=args.dry_run,
    )
    if args.json:
        _print_json([r.__dict__ for r in results])
    else:
        mode = "预览 (dry-run)" if args.dry_run else "已写入数据库"
        print(f"=== 轻量场景初筛结果 [{mode}]，共 {len(results)} 个分 P ===")
        for r in results:
            flag = " [LOCKED]" if r.skipped_locked else ""
            print(
                f"- {r.up_name} | {r.bvid} P{r.page} | "
                f"{r.old_scene_type} -> {r.new_scene_type} (conf={r.confidence:.2f}){flag}"
            )
            print(f"  标题: {r.title}")
            print(f"  依据: {'; '.join(r.reasons)}")
    return 0


def cmd_set_scene(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db)
    db = CorpusDB(cfg.db_path)
    n = db.update_scene_type(
        args.bvid,
        args.page,
        args.scene_type,
        confidence=1.0,
        reason=args.reason,
        lock=not args.no_lock,
    )
    print(f"[OK] 已更新 {n} 条视频分 P 的 scene_type={args.scene_type} (lock={not args.no_lock})")
    return 0


def cmd_set_visibility(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db)
    db = CorpusDB(cfg.db_path)
    n = db.update_visibility(args.bvid, args.page, args.visible)
    print(f"[OK] 已更新 {n} 条视频分 P 的 is_visible={args.visible}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db)
    db = CorpusDB(cfg.db_path)
    svc = SearchService(db)
    up_list = [x.strip() for x in args.up.split(",") if x.strip()] if args.up else None
    scene_list = (
        [x.strip() for x in args.scene.split(",") if x.strip()] if args.scene else None
    )
    res = svc.search(
        SearchQuery(
            q=args.q or "",
            up_names=up_list,
            date_from=args.date_from,
            date_to=args.date_to,
            scene_types=scene_list,
            bvid=args.bvid,
            include_hidden=args.include_hidden,
            fuzzy=not args.exact,
            context_lines=args.context,
            sort=args.sort,
            limit=args.limit,
            offset=args.offset,
        )
    )
    if args.export_clips:
        ids = [it["id"] for it in res["items"]]
        manifest = plan_clip_job(db, subtitle_ids=ids)
        out_path = Path(args.export_clips)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[OK] 已导出 {manifest['segment_count']} 段切片任务清单至: {out_path}")

    if args.json:
        _print_json(res)
    else:
        p = res["pagination"]
        print(
            f"=== 检索结果: q={res['query']!r} | 共命中 {p['total']} 条 "
            f"(显示 {p['offset'] + 1}..{p['offset'] + len(res['items'])}) ==="
        )
        for it in res["items"]:
            print(
                f"[#{it['id']}] {it['up_name']} | {it['pub_date']} | "
                f"{it['bvid']} P{it['page']} | {it['start_sec']:.2f}s-{it['end_sec']:.2f}s "
                f"({it['start_hms']}) | scene={it['scene_type']}"
            )
            print(f"    字幕: {it['content']}")
            if it.get("context_before"):
                for c in it["context_before"]:
                    print(f"    [前文 {c['start_sec']:.2f}s] {c['content']}")
            if it.get("context_after"):
                for c in it["context_after"]:
                    print(f"    [后文 {c['start_sec']:.2f}s] {c['content']}")
            print(f"    空降: {it['jump_url']}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db)
    db = CorpusDB(cfg.db_path)
    _print_json(db.get_stats())
    return 0


def cmd_sync_manifest(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db)
    db = CorpusDB(cfg.db_path)
    manifest = build_manifest(db)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    else:
        print(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")))
    return 0


def cmd_export_delta(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db)
    db = CorpusDB(cfg.db_path)
    remote_manifest = {}
    if args.remote_manifest:
        rm_path = Path(args.remote_manifest)
        if rm_path.exists():
            raw = rm_path.read_text(encoding="utf-8").strip()
            if raw:
                remote_manifest = json.loads(raw)
    res = export_delta(db, remote_manifest, args.out)
    _print_json(res)
    return 0


def cmd_apply_delta(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db)
    db = CorpusDB(cfg.db_path)
    res = apply_delta(db, args.delta_file)
    _print_json(res)
    return 0


def cmd_apply_scene_tags(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db)
    db = CorpusDB(cfg.db_path)
    res = apply_scene_tags(db, args.patch_file)
    _print_json(res)
    return 0


def cmd_apply_speakers(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db)
    db = CorpusDB(cfg.db_path)
    res = apply_speakers(db, args.patch_file)
    _print_json(res)
    return 0


def cmd_build_voiceprint_bank(args: argparse.Namespace) -> int:
    from .voiceprint import build_voiceprint_bank_from_cache

    res = build_voiceprint_bank_from_cache()
    _print_json(res)
    return 0


def cmd_run_voiceprint(args: argparse.Namespace) -> int:
    from .voiceprint import run_voiceprint_batch

    cfg = load_config(db_path=args.db)
    db = CorpusDB(cfg.db_path)
    res = run_voiceprint_batch(
        db,
        cfg,
        scope=args.scope,
        limit=args.limit,
        shard_index=args.shard_index,
        shard_total=args.shard_total,
        device=args.device,
        batch_size=args.batch_size,
        num_threads=args.threads,
        prefetch=args.prefetch,
        force=args.force,
    )
    _print_json(res)
    return 0


def cmd_backfill_metrics(args: argparse.Namespace) -> int:
    cfg = load_config(db_path=args.db)
    metrics_db = SiteMetricsDB(cfg.db_path.parent / "site_metrics.db")
    res = metrics_db.import_nginx_access_logs(log_dir=args.log_dir)
    _print_json(res)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .api import create_app

    cfg = load_config(db_path=args.db)
    enable_admin = not args.public_only
    app = create_app(db_path=cfg.db_path, enable_admin=enable_admin)
    host = args.host or cfg.api_host
    port = args.port or cfg.api_port
    if enable_admin:
        print(
            f"[INFO] 本地后端（含调试台与运维接口）已启动: http://{host}:{port}/debug  "
            f"(OpenAPI: http://{host}:{port}/docs)"
        )
    else:
        print(
            f"[INFO] 公网只读后端已启动（已关闭 admin 与调试台）: http://{host}:{port}/docs"
        )
    uvicorn.run(app, host=host, port=port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zhijiang-corpus",
        description="枝江直播回放 AI 字幕语料库与检索工具（第一阶段本地原型）",
    )
    parser.add_argument("--db", default=None, help="SQLite 数据库文件路径（默认读 ZJ_DB_PATH）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-db", help="初始化 SQLite 数据库与 FTS5 索引")
    p_init.set_defaults(func=cmd_init_db)

    p_login = sub.add_parser("login", help="终端扫码登录 B 站并自动保存 Cookie 到 .env")
    p_login.set_defaults(func=cmd_login)

    p_auth = sub.add_parser("check-auth", help="检查当前 B 站 Cookie 登录状态是否有效")
    p_auth.add_argument("--cookie", default=None, help="临时校验指定 Cookie")
    p_auth.set_defaults(func=cmd_check_auth)

    p_cols = sub.add_parser("list-collections", help="查看 6 位成员 B 站空间里的合集/视频列表")
    p_cols.add_argument("up", nargs="?", default="all", help="UP 简称或 all")
    p_cols.set_defaults(func=cmd_list_collections)

    p_demo = sub.add_parser("seed-demo", help="写入演示字幕数据（仅用于离线功能测试）")
    p_demo.set_defaults(func=cmd_seed_demo)

    p_bvid = sub.add_parser("ingest-bvid", help="抓取指定 BV 号的分 P 与 B 站 AI 字幕并增量入库")
    p_bvid.add_argument("bvids", nargs="+", help="一个或多个 BV 号")
    p_bvid.add_argument("--up-name", default=None, help="手动指定归属 UP 简称")
    p_bvid.add_argument("--pages", default=None, help="仅抓取指定分 P（逗号分隔，如 1,2）")
    p_bvid.add_argument("--cookie", default=None, help="临时覆盖 B 站 Cookie")
    p_bvid.add_argument("--force", action="store_true", help="强制覆盖已有分 P 字幕")
    p_bvid.add_argument("--retry-no-sub", action="store_true", help="重试此前标记为无字幕的分 P")
    p_bvid.set_defaults(func=cmd_ingest_bvid)

    p_up = sub.add_parser("ingest-up", help="按 UP 主「直播回放」合集/列表抓取并增量入库")
    p_up.add_argument("up", help="UP 简称（嘉然/贝拉/乃琳/向晚/心宜/思诺）、UID 或 all")
    p_up.add_argument("--limit", type=int, default=5, help="每位 UP 最多抓取场次（默认 5，传 0 抓全量）")
    p_up.add_argument("--source", default="auto", choices=["auto", "series", "season", "space"], help="列表来源（默认 auto 自动使用内置「直播回放」series_id）")
    p_up.add_argument("--series-id", type=int, default=None, help="手动指定 series_id")
    p_up.add_argument("--season-id", type=int, default=None, help="手动指定 season_id")
    p_up.add_argument("--keyword", default="回放", help="仅 source=space 时生效的搜索关键词")
    p_up.add_argument("--title-contains", default=None, help="本地二次标题过滤子串")
    p_up.add_argument("--since", default=None, help="起始发布日期 YYYY-MM-DD")
    p_up.add_argument("--until", default=None, help="截止发布日期 YYYY-MM-DD")
    p_up.add_argument("--cookie", default=None, help="临时覆盖 B 站 Cookie")
    p_up.add_argument("--force", action="store_true", help="强制覆盖已有字幕")
    p_up.add_argument("--retry-no-sub", action="store_true", help="重试此前无字幕场次")
    p_up.set_defaults(func=cmd_ingest_up)

    p_loc = sub.add_parser("ingest-local", help="导入本地 B 站 BCC JSON 或 SRT 字幕文件")
    p_loc.add_argument("file", help="本地 .json 或 .srt 文件路径")
    p_loc.add_argument("--up-name", required=True, help="UP 主简称（如 嘉然）")
    p_loc.add_argument("--bvid", required=True, help="对应 BV 号")
    p_loc.add_argument("--page", type=int, default=1, help="分 P 号（默认 1）")
    p_loc.add_argument("--pub-date", required=True, help="发布日期 YYYY-MM-DD")
    p_loc.add_argument("--title", default="", help="视频标题")
    p_loc.add_argument("--part-title", default="", help="分 P 标题")
    p_loc.add_argument("--scene-type", default="unknown", choices=VALID_SCENE_TYPES)
    p_loc.add_argument("--force", action="store_true", help="强制替换该分 P 已有字幕")
    p_loc.set_defaults(func=cmd_ingest_local)

    p_tag = sub.add_parser("tag-scenes", help="运行轻量规则场景初筛（单播/双播/团播/联动/特别场次）")
    p_tag.add_argument("--bvid", default=None, help="仅对指定 BV 号运行")
    p_tag.add_argument("--only-unknown", action="store_true", help="仅处理当前为 unknown 的场次")
    p_tag.add_argument("--keep-unknown-solo", action="store_true", help="无多人特征时不自动标 solo，保留 unknown")
    p_tag.add_argument("--dry-run", action="store_true", help="仅预览打标结果，不写入数据库")
    p_tag.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    p_tag.set_defaults(func=cmd_tag_scenes)

    p_set_s = sub.add_parser("set-scene", help="人工修正并锁定指定视频的 scene_type")
    p_set_s.add_argument("bvid", help="BV 号")
    p_set_s.add_argument("scene_type", choices=VALID_SCENE_TYPES, help="目标场景类型")
    p_set_s.add_argument("--page", type=int, default=None, help="指定分 P（不填则更新全部分 P）")
    p_set_s.add_argument("--reason", default="manual_cli", help="标注备注")
    p_set_s.add_argument("--no-lock", action="store_true", help="不锁定（允许后续自动初筛覆盖）")
    p_set_s.set_defaults(func=cmd_set_scene)

    p_vis = sub.add_parser("set-visibility", help="设置视频/分 P 可见性（is_visible=1/0）")
    p_vis.add_argument("bvid", help="BV 号")
    p_vis.add_argument("visible", type=int, choices=[0, 1], help="1=可见，0=隐藏")
    p_vis.add_argument("--page", type=int, default=None, help="指定分 P")
    p_vis.set_defaults(func=cmd_set_visibility)

    p_search = sub.add_parser("search", help="关键词搜索字幕语料")
    p_search.add_argument("q", nargs="?", default="", help="搜索关键词（空格分隔多词 AND）")
    p_search.add_argument("--up", default=None, help="UP 主过滤（可逗号分隔，如 嘉然,贝拉）")
    p_search.add_argument("--date-from", default=None, help="起始日期 YYYY-MM-DD")
    p_search.add_argument("--date-to", default=None, help="结束日期 YYYY-MM-DD")
    p_search.add_argument("--scene", default=None, help="scene_type 过滤（可逗号分隔）")
    p_search.add_argument("--bvid", default=None, help="限定单个 BV 号")
    p_search.add_argument("--context", type=int, default=0, help="显示前后上下文字幕条数")
    p_search.add_argument("--sort", default="date_desc", choices=["date_desc", "date_asc", "relevance"])
    p_search.add_argument("--exact", action="store_true", help="关闭同音/别称模糊扩展，仅严格精确匹配")
    p_search.add_argument("--include-hidden", action="store_true", help="包含隐藏条目")
    p_search.add_argument("--limit", type=int, default=20, help="每页返回条数")
    p_search.add_argument("--offset", type=int, default=0, help="分页偏移量")
    p_search.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    p_search.add_argument("--export-clips", default=None, help="预留：导出 FFmpeg 切片清单 JSON 路径")
    p_search.set_defaults(func=cmd_search)

    p_retry = sub.add_parser("retry-failed", help="仅重试此前抓取失败或待登录的 BV 号")
    p_retry.add_argument("--include-no-sub", action="store_true", help="同时重试此前标记无字幕的场次")
    p_retry.add_argument("--cookie", default=None, help="临时覆盖 B 站 Cookie")
    p_retry.set_defaults(func=cmd_retry_failed)

    p_opt = sub.add_parser("optimize-db", help="批量抓完后压缩优化 FTS5 索引并回收 WAL")
    p_opt.set_defaults(func=cmd_optimize_db)

    p_norm = sub.add_parser(
        "normalize-corpus",
        help="按 config/correction_rules.json 对存量数据库执行枝江专有名词纠错修复（FTS5 加速）",
    )
    p_norm.add_argument("--dry-run", action="store_true", help="仅预览将修正条数与样例，不写库")
    p_norm.add_argument("--batch-size", type=int, default=500, help="每事务更新条数（默认 500）")
    p_norm.set_defaults(func=cmd_normalize_corpus)

    p_asr = sub.add_parser(
        "backfill-asr",
        help="针对无自带字幕的分 P 提取纯音频并运行本地 GPU faster-whisper 转录补录",
    )
    p_asr.add_argument("--bvid", default=None, help="仅补录指定 BV 号")
    p_asr.add_argument("--page", type=int, default=None, help="仅补录指定分 P（配合 --bvid）")
    p_asr.add_argument("--up", default=None, help="仅补录指定 UP 主（如 嘉然）")
    p_asr.add_argument("--limit", type=int, default=5, help="本次最多处理分 P 数（默认 5，0=全部无字幕分 P）")
    p_asr.add_argument("--min-duration", type=int, default=15, help="跳过小于此秒数的碎片分 P（默认 15s）")
    p_asr.add_argument("--max-duration", type=int, default=None, help="仅处理不超过此秒数的分 P")
    p_asr.add_argument("--shortest-first", action="store_true", help="优先处理时长较短的分 P（适合试跑）")
    p_asr.add_argument("--force", action="store_true", help="强制覆盖已有字幕（慎用）")
    p_asr.add_argument(
        "--retranscribe-legacy-asr",
        action="store_true",
        help="仅重录旧版粗断句 asr-large-v3-turbo 分 P 并升级为词级时间戳 asr-v2-large-v3-turbo（支持断点续跑）",
    )
    p_asr.add_argument("--model", default="large-v3-turbo", help="Whisper 模型名（默认 large-v3-turbo）")
    p_asr.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="推理设备")
    p_asr.add_argument("--compute-type", default="int8_float16", help="计算精度（推荐 int8_float16）")
    p_asr.add_argument("--batch-size", type=int, default=12, help="GPU 批处理大小（默认 12）")
    p_asr.add_argument("--beam-size", type=int, default=2, help="解码束宽（默认 2，减少漏标点与乱码）")
    p_asr.add_argument("--chunk-length", type=int, default=10, help="VAD 切片上限秒数（默认 10s，防止合并超长句）")
    p_asr.add_argument("--word-timestamps", action="store_true", help="开启词级时间戳对齐（现已默认开启，保留参数兼容）")
    p_asr.add_argument("--no-word-timestamps", action="store_true", help="关闭词级时间戳（仅极速测试用）")
    p_asr.add_argument("--workers", type=int, default=1, help="单进程内 GPU 推理线程数（默认 1，多并行请用分片多进程）")
    p_asr.add_argument("--prefetch", type=int, default=2, help="后台预取下载+FFmpeg 转码线程数（推荐 1~2）")
    p_asr.add_argument("--shard-index", type=int, default=0, help="多进程分片序号（0 开始）")
    p_asr.add_argument("--shard-total", type=int, default=1, help="多进程分片总数（如 2 表示双进程独立 CUDA 并行）")
    p_asr.add_argument("--hf-mirror", action="store_true", help="使用 hf-mirror.com 镜像下载模型")
    p_asr.add_argument("--keep-audio", action="store_true", help="保留临时音频文件（调试用）")
    p_asr.add_argument("--cookie", default=None, help="临时覆盖 B 站 Cookie")
    p_asr.set_defaults(func=cmd_backfill_asr)

    p_resplit = sub.add_parser(
        "resplit-asr",
        help="对存量 ASR 超长句字幕执行多级智能重断句与异常符号清理（纯 CPU 几秒完成）",
    )
    p_resplit.add_argument("--max-chars", type=int, default=22, help="单行目标字数上限（默认 22 字）")
    p_resplit.add_argument("--dry-run", action="store_true", help="仅预览重断句前后行数，不写库")
    p_resplit.set_defaults(func=cmd_resplit_asr)

    p_stats = sub.add_parser("stats", help="查看当前语料库统计概览")
    p_stats.set_defaults(func=cmd_stats)

    p_manifest = sub.add_parser("sync-manifest", help="导出当前数据库分 P 清单（用于增量同步比对）")
    p_manifest.add_argument("--out", default=None, help="写入指定文件路径（不填则直接输出到 stdout）")
    p_manifest.set_defaults(func=cmd_sync_manifest)

    p_edelta = sub.add_parser("export-delta", help="对比云端清单，仅导出新增或变动的分 P 增量补丁包")
    p_edelta.add_argument("--remote-manifest", required=True, help="从云端拉取的 manifest JSON 文件路径")
    p_edelta.add_argument("--out", required=True, help="输出增量补丁路径（.json.gz）")
    p_edelta.set_defaults(func=cmd_export_delta)

    p_adelta = sub.add_parser("apply-delta", help="在云端 SQLite WAL 模式下热应用增量补丁包（零停服）")
    p_adelta.add_argument("delta_file", help="增量补丁路径（.json.gz）")
    p_adelta.set_defaults(func=cmd_apply_delta)

    p_atags = sub.add_parser("apply-scene-tags", help="在云端热应用纯场景标签轻量补丁（不触碰字幕正文）")
    p_atags.add_argument("patch_file", help="标签补丁路径（.json.gz）")
    p_atags.set_defaults(func=cmd_apply_scene_tags)

    p_aspk = sub.add_parser("apply-speakers", help="在云端热应用纯声纹发言人标注轻量补丁（零 FTS5 重写）")
    p_aspk.add_argument("patch_file", help="声纹补丁路径（.json.gz）")
    p_aspk.set_defaults(func=cmd_apply_speakers)

    p_vp_bank = sub.add_parser("build-voiceprint-bank", help="从 2021~2026 跨时期特征缓存编译正式 6 人声纹模型包")
    p_vp_bank.set_defaults(func=cmd_build_voiceprint_bank)

    p_vp = sub.add_parser("run-voiceprint", help="批量执行声纹发言人标注（默认优先 CUDA GPU 批量推理，支持断点续跑）")
    p_vp.add_argument(
        "--scope",
        default="multi_and_watch",
        choices=["multi_and_watch", "multi_only", "all"],
        help="标注范围（默认 multi_and_watch：双播/团播/特别场 + 影厅/电影反应回）",
    )
    p_vp.add_argument("--limit", type=int, default=0, help="最多处理分 P 数量（0=全部待标注）")
    p_vp.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="推理设备（默认 auto：自动优先 CUDA GPU，不可用时回退 CPU）",
    )
    p_vp.add_argument("--batch-size", type=int, default=64, help="GPU 批量推理 Batch Size（默认 64）")
    p_vp.add_argument("--threads", type=int, default=4, help="CPU 回退模式下的单进程 ONNX 线程数（默认 4）")
    p_vp.add_argument("--prefetch", type=int, default=3, help="后台并发下载+FFmpeg 转码线程数（默认 3）")
    p_vp.add_argument("--shard-index", type=int, default=0, help="多进程分片序号")
    p_vp.add_argument("--shard-total", type=int, default=1, help="多进程分片总数")
    p_vp.add_argument("--force", action="store_true", help="强制重跑已标注分 P")
    p_vp.set_defaults(func=cmd_run_voiceprint)

    p_bmetrics = sub.add_parser("backfill-metrics", help="从 Nginx access.log* 幂等补回历史真实访问人数（UV/PV）")
    p_bmetrics.add_argument("--log-dir", default="/var/log/nginx", help="Nginx 日志目录（默认 /var/log/nginx）")
    p_bmetrics.set_defaults(func=cmd_backfill_metrics)

    p_serve = sub.add_parser("serve", help="启动后端 API 服务（默认本地含调试台，--public-only 为公网只读模式）")
    p_serve.add_argument("--host", default=None, help="监听地址（默认 127.0.0.1）")
    p_serve.add_argument("--port", type=int, default=None, help="监听端口（默认 8000）")
    p_serve.add_argument("--public-only", action="store_true", help="仅暴露公网只读检索 API（模拟云端部署）")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
