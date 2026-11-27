"""早期飞书录播补全 + 二创计画合集缺失补录与场景标签校准脚本.

用法示例：
  # 1. 仅预览（不写数据库），检查将修正的标签与将补录的早期/缺失场次及多P筛选结果
  python scripts/sync_archives.py --dry-run

  # 2. 刷新二创计画 10 个合集最新清单并执行全量同步（标签修正 + 早期补录 + 缺失补录）
  python scripts/sync_archives.py --refresh-erchuang

  # 3. 仅执行其中某一项
  python scripts/sync_archives.py --only-tags
  python scripts/sync_archives.py --only-feishu
  python scripts/sync_archives.py --only-erchuang
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zhijiang_corpus.archive_sync import (
    ERCHUANG_TARGET_SERIES,
    ERCHUANG_UID,
    _video_info_from_cache,
    apply_tag_updates,
    ingest_archive_tasks,
    plan_erchuang_sync,
    plan_feishu_early_sync,
    select_clean_pages,
)
from zhijiang_corpus.bilibili_client import BilibiliClient
from zhijiang_corpus.config import load_config
from zhijiang_corpus.db import CorpusDB


def refresh_erchuang_series_cache(cfg, cache_path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with BilibiliClient(cookie=cfg.bilibili_cookie, delay=0.5, jitter=0.2) as bili:
        for sid, name, _, _ in ERCHUANG_TARGET_SERIES:
            items = bili.list_series_videos(
                mid=ERCHUANG_UID, series_id=sid, limit=None, page_size=100
            )
            out[str(sid)] = {
                "series_id": sid,
                "name": name,
                "count": len(items),
                "videos": [
                    {
                        "bvid": x.bvid,
                        "title": x.title,
                        "pub_ts": x.pub_ts,
                        "pub_date": x.pub_date,
                        "duration_sec": x.duration_sec,
                    }
                    for x in items
                ],
            }
            print(f"[刷新合集] {name} ({sid}): {len(items)} 条", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="补录飞书早期录播与二创计画缺失场次，并校准场景标签")
    parser.add_argument("--db", default=None, help="SQLite 数据库路径（默认读配置）")
    parser.add_argument("--dry-run", action="store_true", help="仅预览规划与多 P 筛选结果，不修改数据库")
    parser.add_argument("--refresh-erchuang", action="store_true", help="重新从 B 站拉取二创计画 10 个合集清单")
    parser.add_argument("--cutoff-date", default="2021-08-19", help="飞书早期录播截止日期（含，默认 2021-08-19）")
    parser.add_argument("--only-tags", action="store_true", help="仅修正二创计画匹配场次的场景标签")
    parser.add_argument("--only-feishu", action="store_true", help="仅补录飞书早期录播")
    parser.add_argument("--only-erchuang", action="store_true", help="仅补录二创计画缺失场次（含标签修正）")
    parser.add_argument("--force", action="store_true", help="强制重抓已存在分 P")
    parser.add_argument("--retry-no-sub", action="store_true", help="重试无字幕分 P")
    parser.add_argument("--delay", type=float, default=None, help="覆盖请求间隔秒数")
    args = parser.parse_args()

    cfg = load_config(db_path=args.db)
    if args.delay is not None:
        cfg.request_delay = args.delay
        cfg.request_jitter = min(cfg.request_jitter, max(0.1, args.delay * 0.4))
    db = CorpusDB(cfg.db_path)
    db.init_db()

    cache_dir = PROJECT_ROOT / "data" / "cache"
    erchuang_series_path = cache_dir / "erchuang_series.json"
    erchuang_infos_path = cache_dir / "erchuang_missing_infos.json"
    feishu_infos_path = cache_dir / "feishu_video_infos.json"

    do_tags = not (args.only_feishu)
    do_feishu = not (args.only_tags or args.only_erchuang)
    do_erchuang = not (args.only_tags or args.only_feishu)

    if args.refresh_erchuang or not erchuang_series_path.exists():
        series_data = refresh_erchuang_series_cache(cfg, erchuang_series_path)
    else:
        series_data = json.loads(erchuang_series_path.read_text(encoding="utf-8"))

    erchuang_infos = (
        json.loads(erchuang_infos_path.read_text(encoding="utf-8"))
        if erchuang_infos_path.exists()
        else {}
    )
    feishu_infos = (
        json.loads(feishu_infos_path.read_text(encoding="utf-8"))
        if feishu_infos_path.exists()
        else {}
    )

    tag_plans, ec_missing_tasks, ec_plan_stats = plan_erchuang_sync(db, series_data)
    feishu_tasks, feishu_plan_stats = plan_feishu_early_sync(
        db, feishu_infos, cutoff_date=args.cutoff_date
    )

    print("=== 规划概览 ===")
    print("飞书早期补录规划:", json.dumps(feishu_plan_stats, ensure_ascii=False))
    print(
        "二创计画对齐规划:",
        json.dumps(
            {k: v for k, v in ec_plan_stats.items() if k != "preserved_tags"},
            ensure_ascii=False,
        ),
    )
    print(f"二创计画保留特例标签数: {len(ec_plan_stats['preserved_tags'])}")
    print("标签变更分布:", Counter((p.old_scene_type, p.new_scene_type) for p in tag_plans))

    if args.dry_run:
        print("\n=== Dry-Run 多 P 筛选抽检 ===")
        for label, tasks, info_cache in (
            ("飞书早期", feishu_tasks, feishu_infos),
            ("二创缺失", ec_missing_tasks, erchuang_infos),
        ):
            mode_cnt = Counter()
            total_sel = 0
            total_drop = 0
            drop_reasons = Counter()
            for t in tasks:
                raw = info_cache.get(t.bvid)
                if not raw or not raw.get("ok"):
                    continue
                vi = _video_info_from_cache(raw)
                sel = select_clean_pages(vi.pages)
                mode_cnt[sel.mode] += 1
                total_sel += len(sel.selected)
                total_drop += len(sel.dropped)
                for _, r in sel.dropped:
                    drop_reasons[r] += 1
            print(
                f"[{label}] 场次={len(tasks)} | 选中分P={total_sel} | 剔除分P={total_drop} "
                f"| 模式={dict(mode_cnt)} | 剔除原因={dict(drop_reasons)}"
            )
        return 0

    if do_tags and tag_plans:
        print(f"\n==> 开始修正 {len(tag_plans)} 场已有视频的场景标签...", flush=True)
        tag_res = apply_tag_updates(db, tag_plans, dry_run=False)
        print("标签修正完成:", json.dumps(tag_res, ensure_ascii=False), flush=True)

    if do_feishu and feishu_tasks:
        print(f"\n==> 开始补录 {len(feishu_tasks)} 场飞书早期录播...", flush=True)
        f_res = ingest_archive_tasks(
            db,
            cfg,
            feishu_tasks,
            video_info_cache=feishu_infos,
            force=args.force,
            retry_no_sub=args.retry_no_sub,
            progress_cb=lambda m: print(m, flush=True),
        )
        print("飞书早期补录完成:", json.dumps({k: v for k, v in f_res.items() if k != "errors"}, ensure_ascii=False))
        if f_res["errors"]:
            print(f"飞书错误数: {len(f_res['errors'])}, 前5条:", f_res["errors"][:5])

    if do_erchuang and ec_missing_tasks:
        print(f"\n==> 开始补录 {len(ec_missing_tasks)} 场二创计画缺失录播...", flush=True)
        e_res = ingest_archive_tasks(
            db,
            cfg,
            ec_missing_tasks,
            video_info_cache=erchuang_infos,
            force=args.force,
            retry_no_sub=args.retry_no_sub,
            progress_cb=lambda m: print(m, flush=True),
        )
        print("二创计画补录完成:", json.dumps({k: v for k, v in e_res.items() if k != "errors"}, ensure_ascii=False))
        if e_res["errors"]:
            print(f"二创错误数: {len(e_res['errors'])}, 前5条:", e_res["errors"][:5])

    print("\n==> 压缩优化 FTS5 索引与 WAL...", flush=True)
    opt = db.optimize_db()
    print("优化完成:", json.dumps(opt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
