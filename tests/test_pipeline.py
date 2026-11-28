"""端到端自动化测试：建库、清洗、增量去重、FTS5 中文检索、场景初筛、切片规划与 FastAPI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from zhijiang_corpus.api import create_app
from zhijiang_corpus.asr_backfill import (
    _is_whisper_hallucination,
    _sanitize_asr_text,
    _split_text_proportionally,
    list_no_subtitle_targets,
    resplit_legacy_asr,
)
from zhijiang_corpus.bilibili_client import SubtitleTrack, VideoInfo, VideoPageInfo
from zhijiang_corpus.cleaner import clean_bcc_payload, clean_srt_text
from zhijiang_corpus.clip_planner import plan_clip_job
from zhijiang_corpus.config import load_config
from zhijiang_corpus.db import CorpusDB
from zhijiang_corpus.ingest import ingest_bvids, ingest_local_file, seed_demo_data
from zhijiang_corpus.normalizer import (
    apply_domain_corrections,
    build_fuzzy_fts_query,
    run_corpus_normalization,
)
from zhijiang_corpus.scene_tagger import run_scene_tagging
from zhijiang_corpus.search import SearchQuery, SearchService
from zhijiang_corpus.wbi import extract_wbi_keys, sign_wbi_params


def test_wbi_signing_deterministic() -> None:
    nav = {
        "wbi_img": {
            "img_url": "https://i0.hdslb.com/bfs/wbi/7cd084941338484aae1ad9425b84077c.png",
            "sub_url": "https://i0.hdslb.com/bfs/wbi/4932caff0ff746eab6f01bf08b70ac45.png",
        }
    }
    img_key, sub_key = extract_wbi_keys(nav)
    assert img_key == "7cd084941338484aae1ad9425b84077c"
    assert sub_key == "4932caff0ff746eab6f01bf08b70ac45"
    signed = sign_wbi_params(
        {"bvid": "BV1xx411c7mD", "cid": 123456, "note": "a!b'c(d)e*"},
        img_key,
        sub_key,
        timestamp=1700000000,
    )
    assert signed["wts"] == "1700000000"
    assert signed["note"] == "abcde"
    assert len(signed["w_rid"]) == 32


def test_cleaner_bcc_and_srt() -> None:
    raw_bcc = {
        "body": [
            {"from": 10.126, "to": 12.5, "content": "  嘉然\u200b今天  吃什么 &amp; 唱歌  "},
            {"from": 10.13, "to": 12.5, "content": "嘉然今天 吃什么 & 唱歌"},  # 完全重复去重
            {"from": 13.0, "to": 14.0, "content": "[音乐]"},
            {"from": 15.0, "to": 14.0, "content": "..."},  # 纯标点噪声
            {"from": -2.0, "to": -1.0, "content": "负时间校正"},
        ]
    }
    cleaned = clean_bcc_payload(raw_bcc)
    assert len(cleaned) == 2
    assert cleaned[0].start_sec == 0.0
    assert cleaned[0].end_sec == 0.0
    assert cleaned[0].content == "负时间校正"
    assert cleaned[1].start_sec == 10.13
    assert cleaned[1].content == "嘉然今天 吃什么 & 唱歌"

    # 校验繁转简归一化
    trad = clean_bcc_payload({"body": [{"from": 1.0, "to": 2.0, "content": "元宵節快樂，看我手語！"}]})
    assert trad[0].content == "元宵节快乐，看我手语！"

    srt = (
        "1\n00:01:02,500 --> 00:01:05,250\n你好，枝江！\n\n"
        "2\n00:01:06,000 --> 00:01:08,000\n（掌声）\n"
    )
    srt_items = clean_srt_text(srt)
    assert len(srt_items) == 1
    assert srt_items[0].start_sec == 62.5
    assert srt_items[0].end_sec == 65.25
    assert srt_items[0].content == "你好，枝江！"


def test_full_pipeline_search_tagging_dedup_and_clips(tmp_path: Path) -> None:
    db_file = tmp_path / "test_corpus.db"
    cfg = load_config(db_path=db_file)
    db = CorpusDB(db_file)

    # 1. 首次写入 demo 数据 + 重复写入验证去重
    first = seed_demo_data(db)
    assert first["videos"] == 6
    assert first["lines_inserted"] > 20
    assert first["lines_duplicate"] == 0

    second = seed_demo_data(db)
    assert second["lines_inserted"] == 0
    assert second["lines_duplicate"] == first["lines_inserted"]

    # 2. 轻量场景初筛（验证标题无法区分的贝拉+向晚双播也能被字幕识别！）
    tagged = run_scene_tagging(db, cfg, dry_run=False)
    by_bvid = {r.bvid: r for r in tagged}
    assert by_bvid["BV1DEMO000001"].new_scene_type == "solo"     # 嘉然单播
    assert by_bvid["BV1DEMO000002"].new_scene_type == "duo"      # 贝拉+向晚（纯靠字幕识别）
    assert by_bvid["BV1DEMO000003"].new_scene_type == "group"    # 乃琳团播
    assert by_bvid["BV1DEMO000004"].new_scene_type == "solo"     # 向晚单播
    assert by_bvid["BV1DEMO000005"].new_scene_type == "duo"      # 心宜+思诺小心思
    assert by_bvid["BV1DEMO000006"].new_scene_type == "special"  # 思诺生日会

    # 3. 人工锁定后再次初筛不会被覆盖
    db.update_scene_type("BV1DEMO000002", 1, "group", reason="人工复核", lock=True)
    retagged = run_scene_tagging(db, cfg, dry_run=False)
    by_bvid2 = {r.bvid: r for r in retagged}
    assert by_bvid2["BV1DEMO000002"].skipped_locked is True
    assert db.get_video("BV1DEMO000002", 1)["scene_type"] == "group"

    # 4. 中文 1字 / 2字 / 长短语 / 英文 / 多关键词 / 过滤 / 分页 / 上下文检索
    svc = SearchService(db)

    # 3字短语 + 跨 UP 命中
    r_essay = svc.search(SearchQuery(q="小作文"))
    assert r_essay["pagination"]["total"] == 5

    # 2字中文精确检索 + UP 过滤
    r_po = svc.search(SearchQuery(q="破防", up_names=["向晚"]))
    assert r_po["pagination"]["total"] == 1
    assert r_po["items"][0]["bvid"] == "BV1DEMO000004"

    # 1字中文检索
    r_one = svc.search(SearchQuery(q="笑", limit=3, offset=0))
    assert r_one["pagination"]["total"] >= 3
    assert len(r_one["items"]) == 3
    assert r_one["pagination"]["has_more"] is True

    # 英文混排检索
    r_en = svc.search(SearchQuery(q="A-SOUL"))
    assert r_en["pagination"]["total"] == 1
    assert r_en["items"][0]["up_name"] == "乃琳"

    # 多词 AND + 日期过滤 + scene_type 过滤 + 上下文
    r_multi = svc.search(
        SearchQuery(
            q="卧底 然然",
            date_from="2024-08-01",
            date_to="2024-08-31",
            scene_types=["group"],
            context_lines=1,
        )
    )
    assert r_multi["pagination"]["total"] == 1
    hit = r_multi["items"][0]
    assert len(hit["context_before"]) == 1
    assert len(hit["context_after"]) == 1

    # 软隐藏 is_visible = 0 默认不可见，include_hidden=True 可见
    db.update_visibility("BV1DEMO000004", 1, 0)
    assert svc.search(SearchQuery(q="破防", up_names=["向晚"]))["pagination"]["total"] == 0
    assert (
        svc.search(SearchQuery(q="破防", up_names=["向晚"], include_hidden=True))[
            "pagination"
        ]["total"]
        == 1
    )

    # 5. FFmpeg 切片清单规划（相邻两句自动合并为一段）
    win = svc.get_subtitle_window("BV1DEMO000001", 1)
    first_two_ids = [win["items"][0]["id"], win["items"][1]["id"]]
    manifest = plan_clip_job(
        db,
        subtitle_ids=first_two_ids,
        pad_before_sec=1.0,
        pad_after_sec=1.0,
        merge_gap_sec=3.0,
    )
    assert manifest["segment_count"] == 1
    assert manifest["segments"][0]["subtitle_ids"] == first_two_ids
    assert manifest["segments"][0]["ffmpeg_cut_cmd"][0] == "ffmpeg"


def test_local_import_and_mocked_bvid_incremental_ingest(tmp_path: Path) -> None:
    db_file = tmp_path / "ingest.db"
    cfg = load_config(db_path=db_file)
    db = CorpusDB(db_file)

    # 本地 BCC JSON 导入
    bcc_path = tmp_path / "sample_bcc.json"
    bcc_path.write_text(
        json.dumps(
            {
                "body": [
                    {"from": 1.0, "to": 3.5, "content": "欢迎来到测试直播回放"},
                    {"from": 4.0, "to": 6.5, "content": "今天状态很不错"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    ins, dup = ingest_local_file(
        db,
        bcc_path,
        up_name="嘉然",
        bvid="BV1LOCAL00001",
        page=1,
        pub_date="2025-02-01",
        title="本地测试回放",
    )
    assert (ins, dup) == (2, 0)

    # 模拟 B站客户端抓取多 P 视频并验证增量跳过
    class FakeBiliClient:
        has_sessdata = True

        def get_video_info(self, bvid: str) -> VideoInfo:
            return VideoInfo(
                bvid=bvid,
                aid=99999,
                title="【直播回放】模拟抓取测试",
                desc="测试简介",
                up_uid=672328094,
                raw_up_name="嘉然今天吃什么",
                pub_ts=1739500000,
                pub_date="2025-02-14",
                duration_sec=3600,
                pages=[
                    VideoPageInfo(page=1, cid=111, part_title="P1 上半场", duration_sec=1800),
                    VideoPageInfo(page=2, cid=222, part_title="P2 下半场", duration_sec=1800),
                ],
            )

        def list_subtitle_tracks(self, bvid: str, cid: int) -> List[SubtitleTrack]:
            if cid == 111:
                return [
                    SubtitleTrack(
                        lan="ai-zh",
                        lan_doc="中文（自动生成）",
                        subtitle_url="https://example.com/p1.json",
                        ai_type=1,
                    )
                ]
            return []  # P2 模拟无字幕

        @staticmethod
        def pick_subtitle_track(tracks: Any, prefer_lans: Any = None) -> Any:
            return tracks[0] if tracks else None

        def fetch_subtitle_json(self, subtitle_url: str) -> Dict[str, Any]:
            return {
                "body": [
                    {"from": 10.0, "to": 13.0, "content": "情人节快乐嘉心糖！"},
                    {"from": 14.0, "to": 17.5, "content": "我们一起唱情歌吧！"},
                ]
            }

    fake = FakeBiliClient()
    s1 = ingest_bvids(db, cfg, ["BV1MOCK000001"], client=fake)  # type: ignore[arg-type]
    assert s1.parts_ok == 1
    assert s1.parts_no_subtitle == 1
    assert s1.lines_inserted == 2

    # 再次增量运行：整部 BV 已被 is_bvid_completed 秒级预判跳过，不重复发任何网络请求
    s2 = ingest_bvids(db, cfg, ["BV1MOCK000001"], client=fake)  # type: ignore[arg-type]
    assert s2.parts_skipped == 1
    assert s2.lines_inserted == 0

    # 若无字幕分 P 已被 Whisper 判定为 asr_empty，后续增量抓取不得将其重置为 no_subtitle 或重复转录
    with db.session() as conn:
        conn.execute(
            "UPDATE videos SET subtitle_status = 'asr_empty' WHERE bvid = 'BV1MOCK000001' AND page = 2"
        )
    assert db.is_bvid_completed("BV1MOCK000001", allow_no_subtitle=True) is True
    assert db.is_bvid_completed("BV1MOCK000001", allow_no_subtitle=False) is True
    s3 = ingest_bvids(db, cfg, ["BV1MOCK000001"], client=fake)  # type: ignore[arg-type]
    assert s3.parts_skipped == 1
    assert db.get_video("BV1MOCK000001", 2)["subtitle_status"] == "asr_empty"

    assert db.optimize_db()["status"] == "ok"


def test_fastapi_endpoints_public_and_admin_separation(tmp_path: Path) -> None:
    db_file = tmp_path / "api.db"
    db = CorpusDB(db_file)
    seed_demo_data(db)
    cfg = load_config(db_path=db_file)
    run_scene_tagging(db, cfg, dry_run=False)

    # 1. 云端公网只读模式（enable_admin=False）：不暴露任何 admin 接口或调试页
    pub_app = create_app(db_path=db_file, enable_admin=False)
    pub_client = TestClient(pub_app)
    assert pub_client.get("/api/v1/health").status_code == 200
    assert pub_client.get("/api/v1/members").status_code == 200
    assert pub_client.get("/debug").status_code == 404
    assert pub_client.get("/api/v1/admin/auth/status").status_code == 404
    assert pub_client.get("/api/v1/admin/cloud/config").status_code == 404
    assert pub_client.get("/api/v1/admin/announcements").status_code == 404
    assert pub_client.get("/api/v1/admin/asr/status").status_code == 404
    assert pub_client.get("/api/v1/admin/voiceprint/status").status_code == 404
    assert pub_client.patch(
        "/api/v1/admin/videos/BV1DEMO000001/scene",
        json={"page": 1, "scene_type": "special"},
    ).status_code == 404

    st = pub_client.get("/api/v1/stats").json()
    assert "db_path" not in st  # 公网统计不暴露服务器本地磁盘路径
    assert st["video_parts_total"] == 6
    assert st["subtitles_total"] > 20
    assert st["latest_video"] is not None
    assert st["latest_video"]["bvid"].startswith("BV1")
    assert st["latest_video"]["video_url"].startswith("https://www.bilibili.com/video/")
    assert isinstance(st["announcements"], list)
    assert "visitors" in st and st["visitors"]["total_uv"] == 0
    # 校验访问统计：同一浏览器二次访问 UV 去重、PV 累加，爬虫不计入
    st_v1 = pub_client.get(
        "/api/v1/stats?track=1",
        headers={"User-Agent": "Mozilla/5.0 Chrome/120.0", "X-Forwarded-For": "203.0.113.10"},
    ).json()
    assert st_v1["visitors"]["total_uv"] == 1
    assert st_v1["visitors"]["total_pv"] == 1
    st_v2 = pub_client.get(
        "/api/v1/stats?track=1",
        headers={"User-Agent": "Mozilla/5.0 Chrome/120.0", "X-Forwarded-For": "203.0.113.10"},
    ).json()
    assert st_v2["visitors"]["total_uv"] == 1
    assert st_v2["visitors"]["total_pv"] == 2
    st_bot = pub_client.get(
        "/api/v1/stats?track=1",
        headers={"User-Agent": "Googlebot/2.1", "X-Forwarded-For": "203.0.113.99"},
    ).json()
    assert st_bot["visitors"]["total_uv"] == 1
    assert st_bot["visitors"]["total_pv"] == 2

    # 校验从 Nginx 历史日志幂等补回访问人数
    from zhijiang_corpus.metrics import SiteMetricsDB

    ng_dir = tmp_path / "nginx_logs"
    ng_dir.mkdir()
    (ng_dir / "access.log").write_text(
        '198.51.100.1 - - [26/Nov/2026:12:00:00 +0000] "GET / HTTP/2.0" 200 4096 "-" "Mozilla/5.0 Chrome/120"\n'
        '198.51.100.1 - - [26/Nov/2026:12:05:00 +0000] "GET / HTTP/2.0" 200 4096 "-" "Mozilla/5.0 Chrome/120"\n'
        '198.51.100.2 - - [26/Nov/2026:13:00:00 +0000] "GET / HTTP/2.0" 200 4096 "-" "Mozilla/5.0 Safari/605"\n'
        '198.51.100.3 - - [26/Nov/2026:13:10:00 +0000] "GET /api/v1/stats HTTP/2.0" 200 500 "-" "Mozilla/5.0"\n'
        '198.51.100.4 - - [26/Nov/2026:13:20:00 +0000] "GET / HTTP/2.0" 200 4096 "-" "Googlebot/2.1"\n',
        encoding="utf-8",
    )
    mdb = SiteMetricsDB(tmp_path / "ng_metrics.db")
    bf1 = mdb.import_nginx_access_logs(log_dir=ng_dir)
    assert bf1["hits_matched"] == 3
    assert bf1["visitors"]["total_uv"] == 2
    assert bf1["visitors"]["total_pv"] == 3
    # 重复执行一次必须幂等，不翻倍
    bf2 = mdb.import_nginx_access_logs(log_dir=ng_dir)
    assert bf2["visitors"]["total_uv"] == 2
    assert bf2["visitors"]["total_pv"] == 3
    ann_res = pub_client.get("/api/v1/announcements")
    assert ann_res.status_code == 200
    assert isinstance(ann_res.json()["items"], list)
    assert len(ann_res.json()["items"]) >= 1

    sr = pub_client.get("/api/v1/search", params={"q": "小作文", "up_name": "嘉然,向晚"}).json()
    assert sr["pagination"]["total"] == 2
    # 校验规范化的 B 站空降 URL（带尾部斜杠、单 P 不带冗余 p=1）
    first_item = sr["items"][0]
    assert first_item["jump_url"].startswith(f"https://www.bilibili.com/video/{first_item['bvid']}/?t=")
    assert first_item["video_url"] == f"https://www.bilibili.com/video/{first_item['bvid']}/"

    # 2. 本地运维模式（enable_admin=True）：允许调用 /api/v1/admin/* 与 /debug
    loc_app = create_app(db_path=db_file, enable_admin=True)
    loc_client = TestClient(loc_app)
    assert loc_client.get("/debug").status_code == 200

    patch = loc_client.patch(
        "/api/v1/admin/videos/BV1DEMO000001/scene",
        json={"page": 1, "scene_type": "special", "lock": True, "reason": "api_test"},
    )
    assert patch.status_code == 200

    sr2 = loc_client.get("/api/v1/search", params={"q": "小作文", "scene_type": "special"}).json()
    assert any(it["bvid"] == "BV1DEMO000001" for it in sr2["items"])

    # 校验【突击】单播优先规则（排除双人/多人联机突击）
    from zhijiang_corpus.scene_tagger import classify_video_scene

    tuji_solo = classify_video_scene(
        {
            "bvid": "BV1TUJI00001",
            "page": 1,
            "up_name": "嘉然",
            "title": "【直播回放】【突击】一起看叶问2！ 2026年09月05日22点场",
            "part_title": "",
            "scene_type": "unknown",
            "extra_meta": "{}",
        },
        [(10.0, "贝拉和乃琳上次说的这个电影真好看"), (700.0, "贝拉乃琳")],
        cfg.members,
    )
    assert tuji_solo.new_scene_type == "solo"
    tuji_duo = classify_video_scene(
        {
            "bvid": "BV1TUJI00002",
            "page": 1,
            "up_name": "乃琳",
            "title": "【直播回放】【突击】和然比一起生化9！ 2026年03月10日18点场",
            "part_title": "",
            "scene_type": "unknown",
            "extra_meta": "{}",
        },
        [(10.0, "然比快来")],
        cfg.members,
    )
    assert tuji_duo.new_scene_type == "duo"

    hit_id = sr["items"][0]["id"]
    plan = loc_client.post("/api/v1/admin/clips/plan", json={"subtitle_ids": [hit_id]}).json()
    assert plan["segment_count"] == 1
    assert plan["segments"][0]["subtitle_ids"] == [hit_id]

    # 3. 本地运维中心接口：场景锁定汇总 + ASR 进度 + 纯标签轻量补丁导出与热应用
    sc_sum = loc_client.get("/api/v1/admin/scenes/summary").json()
    assert sc_sum["total_locked_bvids"] >= 1
    asr_st = loc_client.get("/api/v1/admin/asr/status").json()
    assert "progress" in asr_st and "job" in asr_st

    from zhijiang_corpus.sync_delta import apply_scene_tags, export_scene_tags

    cloud_db_file = tmp_path / "cloud_tags.db"
    cloud_db = CorpusDB(cloud_db_file)
    cloud_db.init_db()
    seed_demo_data(cloud_db)
    patch_file = tmp_path / "scene_tags.json.gz"
    exp_tag = export_scene_tags(db, patch_file, only_locked=True)
    assert exp_tag["parts_exported"] >= 1
    ap_tag = apply_scene_tags(cloud_db, patch_file)
    assert ap_tag["parts_updated"] == 1
    assert cloud_db.get_video("BV1DEMO000001", 1)["scene_type"] == "special"
    assert cloud_db.get_video("BV1DEMO000001", 1)["scene_locked"] == 1

    # 4. 声纹发言人先验、分桶对齐、精准发言人过滤、以及纯声纹云端轻量补丁校验
    from zhijiang_corpus.sync_delta import apply_speakers, export_speakers
    from zhijiang_corpus.voiceprint import _snap_audio_bounds, active_candidates_for_video

    assert "心宜" not in active_candidates_for_video(up_name="嘉然", pub_date="2022-08-01", scene_type="group")
    assert "向晚" not in active_candidates_for_video(up_name="嘉然", pub_date="2026-08-01", scene_type="group")
    assert active_candidates_for_video(up_name="贝拉", pub_date="2026-11-01", scene_type="solo") == ["贝拉"]
    assert active_candidates_for_video(up_name="心宜", pub_date="2026-08-09", scene_type="duo", title="【双人电台】") == ["心宜", "思诺"]
    s0, e0 = _snap_audio_bounds(10.0, 10.95, 320000, 16000)
    assert e0 - s0 == 15360  # 0.95s 精确对齐到 96 帧桶 (0.96s)

    vp_st = loc_client.get("/api/v1/admin/voiceprint/status").json()
    assert "progress" in vp_st and "job" in vp_st
    assert "cuda_available" in vp_st["progress"]

    # 模拟本地写入声纹标注（含双人重叠与 其他/原声），验证 speaker_filter 跨频道检索与轻量同步补丁
    with db.session() as conn:
        sub_ids = [r[0] for r in conn.execute("SELECT id FROM subtitles WHERE bvid='BV1DEMO000001' ORDER BY start_sec").fetchall()]
        conn.execute("UPDATE subtitles SET speaker_name='嘉然+乃琳', speaker_conf=0.88 WHERE id=?", (sub_ids[0],))
        if len(sub_ids) > 1:
            conn.execute("UPDATE subtitles SET speaker_name='其他/原声', speaker_conf=0.91 WHERE id=?", (sub_ids[1],))
        conn.execute("UPDATE videos SET voiceprint_status='ok' WHERE bvid='BV1DEMO000001' AND page=1")

    sr_sp = loc_client.get("/api/v1/search", params={"bvid": "BV1DEMO000001", "up_name": "乃琳", "speaker_filter": "true"}).json()
    assert sr_sp["pagination"]["total"] == 1
    assert sr_sp["items"][0]["speaker_name"] == "嘉然+乃琳"

    sp_patch = tmp_path / "speakers.json.gz"
    exp_sp = export_speakers(db, sp_patch)
    assert exp_sp["parts_exported"] == 1
    ap_sp = apply_speakers(cloud_db, sp_patch)
    assert ap_sp["parts_applied"] == 1
    assert cloud_db.get_video("BV1DEMO000001", 1)["voiceprint_status"] == "ok"


def test_asr_backfill_helpers(tmp_path: Path) -> None:
    db_file = tmp_path / "asr_helper.db"
    db = CorpusDB(db_file)
    db.init_db()
    from zhijiang_corpus.db import VideoRecord

    # 写入一个 3 秒碎片和一个 600 秒正常无字幕分 P
    db.save_video_and_subtitles(
        VideoRecord(
            bvid="BV1SHORT0001",
            page=1,
            cid=1,
            up_name="心宜",
            pub_date="2026-01-01",
            duration_sec=3,
            subtitle_status="no_subtitle",
        ),
        [],
    )
    db.save_video_and_subtitles(
        VideoRecord(
            bvid="BV1LONG00001",
            page=1,
            cid=2,
            up_name="心宜",
            pub_date="2026-01-02",
            duration_sec=600,
            subtitle_status="no_subtitle",
        ),
        [],
    )
    targets = list_no_subtitle_targets(db, min_duration=15, limit=10)
    assert [t["bvid"] for t in targets] == ["BV1LONG00001"]
    assert _is_whisper_hallucination("以下是B站虚拟偶像嘉然") is True
    assert _is_whisper_hallucination("今天晚上吃什么呀") is False

    # 校验 ASR 异常标点清洗与多级断句切分
    assert _sanitize_asr_text("我们去玩吧﹐然后回家﹒") == "我们去玩吧，然后回家。"
    long_text = (
        "今天晚上大家一起来枝江看直播吧，我觉得这个新衣服真的特别好看，"
        "而且背景音乐也非常带感，你们觉得呢？如果喜欢的话记得点个关注呀！"
    )
    pieces = _split_text_proportionally(10.0, 40.0, long_text, max_chars=22)
    assert len(pieces) >= 4
    assert all(len(text) <= 28 for _, _, text in pieces)
    assert pieces[0][0] == pytest.approx(10.0, abs=0.05)
    assert pieces[-1][1] == pytest.approx(40.0, abs=0.05)
    for i in range(len(pieces) - 1):
        assert pieces[i][0] <= pieces[i][1] <= pieces[i + 1][0] + 0.01

    # 无标点超长连续口语也必须强制软切分
    unpunct = "其实我今天本来是想早点起床准备直播的但是闹钟响了之后我又赖床了半个小时结果差点迟到"
    pieces_u = _split_text_proportionally(0.0, 20.0, unpunct, max_chars=22)
    assert len(pieces_u) >= 2
    assert all(len(text) <= 28 for _, _, text in pieces_u)

    # 校验 resplit_legacy_asr 仅重切旧版 ASR 且保留 asr-large-v3-turbo 标记供 GPU 重录识别
    from zhijiang_corpus.db import SubtitleRow

    db.save_video_and_subtitles(
        VideoRecord(
            bvid="BV1LEGACY001",
            page=1,
            cid=3,
            up_name="嘉然",
            pub_date="2026-01-03",
            duration_sec=120,
            subtitle_status="ok",
            subtitle_lan="asr-large-v3-turbo",
        ),
        [
            SubtitleRow(
                up_name="嘉然",
                bvid="BV1LEGACY001",
                page=1,
                pub_date="2026-01-03",
                start_sec=0.0,
                end_sec=30.0,
                content=long_text,
                source="asr-large-v3-turbo",
            )
        ],
    )
    db.save_video_and_subtitles(
        VideoRecord(
            bvid="BV1NEWASR001",
            page=1,
            cid=4,
            up_name="嘉然",
            pub_date="2026-01-04",
            duration_sec=120,
            subtitle_status="ok",
            subtitle_lan="asr-v2-large-v3-turbo",
        ),
        [
            SubtitleRow(
                up_name="嘉然",
                bvid="BV1NEWASR001",
                page=1,
                pub_date="2026-01-04",
                start_sec=0.0,
                end_sec=3.0,
                content="短句无需重切",
                source="asr-v2-large-v3-turbo",
            )
        ],
    )
    db.save_video_and_subtitles(
        VideoRecord(
            bvid="BV1OFFICIAL1",
            page=1,
            cid=5,
            up_name="嘉然",
            pub_date="2026-01-05",
            duration_sec=120,
            subtitle_status="ok",
            subtitle_lan="ai-zh",
        ),
        [
            SubtitleRow(
                up_name="嘉然",
                bvid="BV1OFFICIAL1",
                page=1,
                pub_date="2026-01-05",
                start_sec=0.0,
                end_sec=3.0,
                content="官方字幕不动",
                source="ai-zh",
            )
        ],
    )

    res = resplit_legacy_asr(db, max_chars=22, dry_run=False)
    assert res["parts_modified"] == 1
    assert res["rows_after"] > res["rows_before"]

    with db.connect() as conn:
        vrow = conn.execute(
            "SELECT subtitle_lan, line_count FROM videos WHERE bvid='BV1LEGACY001' AND page=1"
        ).fetchone()
        assert vrow["subtitle_lan"] == "asr-large-v3-turbo"
        assert vrow["line_count"] >= 4

    # 校验 --retranscribe-legacy-asr 目标筛选只选中旧版 asr-large-v3-turbo，跳过 v2 与官方字幕
    legacy_targets = list_no_subtitle_targets(db, min_duration=15, limit=10, retranscribe_legacy_asr=True)
    assert [t["bvid"] for t in legacy_targets] == ["BV1LEGACY001"]


def test_domain_normalization_and_fuzzy_search(tmp_path: Path) -> None:
    # 1. Tier 1 & Tier 2 纠错与排除保护校验
    assert apply_domain_corrections("欢迎 a sou 和佳然、欣怡、奶麒麟、顶晚人、星球仪！") == (
        "欢迎 A-SOUL 和嘉然、心宜、奶淇琳、顶碗人、心球仪！"
    )
    # 保留 小海螺 / 小螺，排除 斯诺克 / 家乐福 / 不是心仪 / 心仪的 / 一心一意 / 小心一点
    assert apply_domain_corrections("欢迎小海螺和小螺，我喜欢打斯诺克，也喜欢斯诺！") == (
        "欢迎小海螺和小螺，我喜欢打斯诺克，也喜欢思诺！"
    )
    assert apply_domain_corrections("心宜不是心仪，送你心仪的礼物，欢迎心仪公主") == (
        "心宜不是心仪，送你心仪的礼物，欢迎心宜公主"
    )
    assert apply_domain_corrections("一心一意小心一点，乃零三分，陪诺零点庆生") == (
        "一心一意小心一点，乃0三分，陪诺零点庆生"
    )
    assert apply_domain_corrections("去家乐福买东西，还有家乐", up_name="嘉然", pub_date="2022-03-01") == "去家乐福买东西，还有珈乐"
    assert apply_domain_corrections("去农家乐，参加乐队比赛，大家乐意", up_name="嘉然", pub_date="2022-03-01") == "去农家乐，参加乐队比赛，大家乐意"
    assert apply_domain_corrections("2022年5月10日后不再替换嘉乐和家乐", up_name="乃琳", pub_date="2022-05-11") == "2022年5月10日后不再替换嘉乐和家乐"
    assert apply_domain_corrections("思诺去农家乐参加乐队", up_name="思诺", pub_date="2026-10-15") == "思诺去农家乐参加乐队"
    assert apply_domain_corrections("我是一宝", up_name="心宜") == "我是宜宝"
    assert apply_domain_corrections("我是一宝", up_name="嘉然") == "我是一宝"

    # 2. 同音模糊查询表达式校验
    expr, expanded = build_fuzzy_fts_query("小海诺", fuzzy=True)
    assert '"小 海 诺"' in expr and '"小 海 螺"' in expr and '"小 螺"' in expr
    assert "小海诺" in expanded

    # 3. 存量批量修复 + 模糊搜索端到端校验
    db_file = tmp_path / "norm.db"
    db = CorpusDB(db_file)
    db.init_db()
    from zhijiang_corpus.db import SubtitleRow, VideoRecord

    db.save_video_and_subtitles(
        VideoRecord(bvid="BV1NORM00001", page=1, up_name="心宜", pub_date="2026-01-01", subtitle_status="ok"),
        [
            SubtitleRow(up_name="心宜", bvid="BV1NORM00001", page=1, pub_date="2026-01-01", start_sec=1.0, end_sec=3.0, content="欣怡和斯诺欢迎奶麒麟和小海螺"),
            SubtitleRow(up_name="心宜", bvid="BV1NORM00001", page=1, pub_date="2026-01-01", start_sec=4.0, end_sec=6.0, content="心宜不是心仪哦"),
        ],
    )
    # 模拟历史遗留未清洗入库后执行批量修复
    res = run_corpus_normalization(db, dry_run=False)
    assert res["updated"] == 1

    svc = SearchService(db)
    # 搜 小海诺（模糊模式）能同时搜到保留原文的 小海螺！
    r_fuzzy = svc.search(SearchQuery(q="小海诺", fuzzy=True))
    assert r_fuzzy["pagination"]["total"] == 1
    assert "小海螺" in r_fuzzy["items"][0]["content"]
    # 精确模式不扩展
    r_exact = svc.search(SearchQuery(q="小海诺", fuzzy=False))
    assert r_exact["pagination"]["total"] == 0
    # 搜 思诺 能搜到被修复后的 斯诺 -> 思诺
    r_sinuo = svc.search(SearchQuery(q="思诺", fuzzy=False))
    assert r_sinuo["pagination"]["total"] == 1


def test_incremental_delta_sync_and_live_stats(tmp_path: Path) -> None:
    from zhijiang_corpus.db import SubtitleRow, VideoRecord
    from zhijiang_corpus.sync_delta import apply_delta, build_manifest, export_delta

    local_db = CorpusDB(tmp_path / "local.db")
    cloud_db = CorpusDB(tmp_path / "cloud.db")
    local_db.init_db()
    cloud_db.init_db()

    # 初始共同数据：旧分 P（2026-11-20）
    v_old = VideoRecord(bvid="BV1SYNC00001", page=1, up_name="嘉然", pub_date="2026-11-20", subtitle_status="ok", scene_type="solo")
    s_old = [SubtitleRow(up_name="嘉然", bvid="BV1SYNC00001", page=1, pub_date="2026-11-20", start_sec=10.0, end_sec=12.0, content="旧直播玉米粑粑")]
    local_db.save_video_and_subtitles(v_old, s_old)
    cloud_db.save_video_and_subtitles(v_old, s_old)

    # 本地新入库 1 场最新直播（2026-11-28），云端尚未同步
    v_new = VideoRecord(bvid="BV1SYNC00002", page=1, up_name="思诺", pub_date="2026-11-28", subtitle_status="ok", scene_type="solo")
    s_new = [
        SubtitleRow(up_name="思诺", bvid="BV1SYNC00002", page=1, pub_date="2026-11-28", start_sec=5.0, end_sec=8.0, content="小海诺增量同步成功啦"),
        SubtitleRow(up_name="思诺", bvid="BV1SYNC00002", page=1, pub_date="2026-11-28", start_sec=9.0, end_sec=12.0, content="首页时间跨度也会自动更新"),
    ]
    local_db.save_video_and_subtitles(v_new, s_new)

    # 1. 拉取云端清单并仅导出增量补丁
    remote_manifest = build_manifest(cloud_db)
    delta_file = tmp_path / "delta.json.gz"
    exp = export_delta(local_db, remote_manifest, delta_file)
    assert exp["parts_new"] == 1
    assert exp["parts_changed"] == 1
    assert exp["subtitles_exported"] == 2

    # 2. 在云端公网只读接口保持挂载的同时热应用补丁
    pub_app = create_app(db_path=cloud_db.db_path, enable_admin=False)
    client = TestClient(pub_app)
    before_stats = client.get("/api/v1/stats").json()
    assert before_stats["date_max"] == "2026-11-20"
    assert before_stats["subtitles_total"] == 1

    res = apply_delta(cloud_db, delta_file)
    assert res["parts_applied"] == 1
    assert res["subtitles_inserted"] == 2

    # 3. 不重启服务，公网 /api/v1/stats 与 /api/v1/search 立刻生效
    after_stats = client.get("/api/v1/stats").json()
    assert after_stats["date_max"] == "2026-11-28"
    assert after_stats["subtitles_total"] == 3
    assert after_stats["video_parts_ok"] == 2

    sr = client.get("/api/v1/search", params={"q": "增量同步成功"}).json()
    assert sr["pagination"]["total"] == 1
    assert sr["items"][0]["bvid"] == "BV1SYNC00002"

    # 4. 再次导出增量应为 0 变动
    exp2 = export_delta(local_db, build_manifest(cloud_db), delta_file)
    assert exp2["parts_changed"] == 0


def test_room_and_speaker_independent_search(tmp_path: Path) -> None:
    """验证「直播间 (up_name)」与「发言人 (speaker)」同级独立组合检索逻辑（单播默认本人）."""
    from zhijiang_corpus.db import SubtitleRow, VideoRecord

    db = CorpusDB(tmp_path / "speaker_search.db")
    db.init_db()

    # 1) 嘉然直播间 - 团播（已声纹标注：嘉然、乃琳、嘉然+乃琳、其他/原声）
    v_group = VideoRecord(
        bvid="BV1SPK000001", page=1, up_name="嘉然", pub_date="2026-11-20",
        subtitle_status="ok", scene_type="group", voiceprint_status="ok",
    )
    s_group = [
        SubtitleRow(up_name="嘉然", bvid="BV1SPK000001", page=1, pub_date="2026-11-20", start_sec=10.0, end_sec=12.0, content="火锅真好吃呀", scene_type="group", speaker_name="嘉然", speaker_conf=0.96),
        SubtitleRow(up_name="嘉然", bvid="BV1SPK000001", page=1, pub_date="2026-11-20", start_sec=20.0, end_sec=22.0, content="火锅我也喜欢吃", scene_type="group", speaker_name="乃琳", speaker_conf=0.95),
        SubtitleRow(up_name="嘉然", bvid="BV1SPK000001", page=1, pub_date="2026-11-20", start_sec=30.0, end_sec=32.0, content="火锅一起吃吧", scene_type="group", speaker_name="嘉然+乃琳", speaker_conf=0.91),
        SubtitleRow(up_name="嘉然", bvid="BV1SPK000001", page=1, pub_date="2026-11-20", start_sec=40.0, end_sec=42.0, content="火锅电影旁白原声", scene_type="group", speaker_name="其他/原声", speaker_conf=0.88),
    ]
    db.save_video_and_subtitles(v_group, s_group)

    # 2) 嘉然直播间 - 单播（未跑声纹，默认发言人为嘉然本人）
    v_jr_solo = VideoRecord(
        bvid="BV1SPK000002", page=1, up_name="嘉然", pub_date="2026-11-21",
        subtitle_status="ok", scene_type="solo", voiceprint_status="",
    )
    db.save_video_and_subtitles(
        v_jr_solo,
        [SubtitleRow(up_name="嘉然", bvid="BV1SPK000002", page=1, pub_date="2026-11-21", start_sec=5.0, end_sec=7.0, content="单播吃火锅咯", scene_type="solo")],
    )

    # 3) 乃琳直播间 - 单播（未跑声纹，默认发言人为乃琳本人）
    v_nl_solo = VideoRecord(
        bvid="BV1SPK000003", page=1, up_name="乃琳", pub_date="2026-11-22",
        subtitle_status="ok", scene_type="solo", voiceprint_status="",
    )
    db.save_video_and_subtitles(
        v_nl_solo,
        [SubtitleRow(up_name="乃琳", bvid="BV1SPK000003", page=1, pub_date="2026-11-22", start_sec=8.0, end_sec=10.0, content="乃琳单播聊火锅", scene_type="solo")],
    )

    pub_app = create_app(db_path=db.db_path, enable_admin=False)
    client = TestClient(pub_app)

    # A. 仅限直播间=嘉然（不过滤发言人）-> 嘉然直播间全部 5 条（含乃琳、合唱、原声、单播）
    r_room = client.get("/api/v1/search", params={"q": "火锅", "up_name": "嘉然"}).json()
    assert r_room["pagination"]["total"] == 5

    # B. 仅限发言人=乃琳（跨全部直播间）-> 乃琳单播(1) + 嘉然团播里乃琳(1) + 嘉然+乃琳合唱(1) = 3 条
    r_sp = client.get("/api/v1/search", params={"q": "火锅", "speaker": "乃琳"}).json()
    assert r_sp["pagination"]["total"] == 3
    assert {it["content"] for it in r_sp["items"]} == {"火锅我也喜欢吃", "火锅一起吃吧", "乃琳单播聊火锅"}

    # C. 指定直播间=嘉然 且 发言人=乃琳 -> 只出嘉然直播间里乃琳说的话（含合唱）= 2 条，不含乃琳自己的单播
    r_both_nl = client.get("/api/v1/search", params={"q": "火锅", "up_name": "嘉然", "speaker": "乃琳"}).json()
    assert r_both_nl["pagination"]["total"] == 2
    assert {it["content"] for it in r_both_nl["items"]} == {"火锅我也喜欢吃", "火锅一起吃吧"}

    # D. 指定直播间=嘉然 且 发言人=嘉然 -> 嘉然团播本人(1) + 嘉然+乃琳(1) + 嘉然单播默认本人(1) = 3 条（排除乃琳独白与电影原声）
    r_both_jr = client.get("/api/v1/search", params={"q": "火锅", "up_name": "嘉然", "speaker": "嘉然"}).json()
    assert r_both_jr["pagination"]["total"] == 3
    assert {it["content"] for it in r_both_jr["items"]} == {"火锅真好吃呀", "火锅一起吃吧", "单播吃火锅咯"}


def test_delete_video_api_reingest_and_delta_sync(tmp_path: Path) -> None:
    """验证已入库场次的按分P/整场删除、FTS5 索引同步、公网隔离、删除后重抓及增量同步云端删除."""
    from zhijiang_corpus.db import SubtitleRow, VideoRecord
    from zhijiang_corpus.sync_delta import apply_delta, build_manifest, export_delta

    db_file = tmp_path / "delete_test.db"
    db = CorpusDB(db_file)
    db.init_db()

    # 写入一个双 P 场次（P1 有字幕，P2 尚未生成 AI 字幕被误导入为 no_subtitle）+ 另一场正常录播
    db.save_video_and_subtitles(
        VideoRecord(
            bvid="BV1DEL000001",
            page=1,
            up_name="嘉然",
            title="误提前导入的回放",
            part_title="P1 上半场",
            pub_date="2026-11-28",
            subtitle_status="ok",
            scene_type="solo",
        ),
        [
            SubtitleRow(
                up_name="嘉然",
                bvid="BV1DEL000001",
                page=1,
                pub_date="2026-11-28",
                start_sec=1.0,
                end_sec=3.0,
                content="待删除测试台词独一无二",
                scene_type="solo",
            )
        ],
    )
    db.save_video_and_subtitles(
        VideoRecord(
            bvid="BV1DEL000001",
            page=2,
            up_name="嘉然",
            title="误提前导入的回放",
            part_title="P2 下半场",
            pub_date="2026-11-28",
            subtitle_status="no_subtitle",
            scene_type="solo",
        ),
        [],
    )
    db.save_video_and_subtitles(
        VideoRecord(
            bvid="BV1KEEP00001",
            page=1,
            up_name="乃琳",
            title="保留的正常场次",
            pub_date="2026-11-27",
            subtitle_status="ok",
            scene_type="solo",
        ),
        [
            SubtitleRow(
                up_name="乃琳",
                bvid="BV1KEEP00001",
                page=1,
                pub_date="2026-11-27",
                start_sec=5.0,
                end_sec=8.0,
                content="保留场次正常字幕",
                scene_type="solo",
            )
        ],
    )
    db.log_ingest("run1", "BV1DEL000001", 2, "no_subtitle", "no track yet")

    # 模拟云端已有相同初始数据
    cloud_db = CorpusDB(tmp_path / "delete_cloud.db")
    cloud_db.init_db()
    init_delta = tmp_path / "init_delta.json.gz"
    export_delta(db, build_manifest(cloud_db), init_delta)
    apply_delta(cloud_db, init_delta)
    assert cloud_db.get_video("BV1DEL000001", 1) is not None
    assert cloud_db.get_video("BV1DEL000001", 2) is not None

    # 1. 公网只读模式禁止删除接口
    pub_client = TestClient(create_app(db_path=db_file, enable_admin=False))
    assert pub_client.delete("/api/v1/admin/videos/BV1DEL000001?page=2").status_code == 404

    # 2. 本地管理模式：支持按 BV/标题筛选 + 删除单分P + 删除整场 BV
    loc_client = TestClient(create_app(db_path=db_file, enable_admin=True))
    vlist = loc_client.get("/api/v1/videos", params={"status": "", "q": "BV1DEL000001"}).json()
    assert vlist["total"] == 2

    # 删除 P2（no_subtitle 分P），P1 保持完好
    r_del_p2 = loc_client.delete("/api/v1/admin/videos/BV1DEL000001", params={"page": 2})
    assert r_del_p2.status_code == 200
    d2 = r_del_p2.json()
    assert d2["deleted_videos"] == 1
    assert d2["deleted_subtitles"] == 0
    assert db.get_video("BV1DEL000001", 2) is None
    assert db.get_video("BV1DEL000001", 1) is not None

    # 删除整个 BV1DEL000001（剩余的 P1 及其字幕和 FTS5 索引一并移除）
    r_del_all = loc_client.delete("/api/v1/admin/videos/BV1DEL000001")
    assert r_del_all.status_code == 200
    da = r_del_all.json()
    assert da["deleted_videos"] == 1
    assert da["deleted_subtitles"] == 1
    assert db.get_video("BV1DEL000001", 1) is None
    assert db.is_bvid_completed("BV1DEL000001", allow_no_subtitle=True) is False
    # FTS5 全文检索已同步清空
    sr = loc_client.get("/api/v1/search", params={"q": "待删除测试台词独一无二"}).json()
    assert sr["pagination"]["total"] == 0
    # 再次删除同一 BV 返回 404
    assert loc_client.delete("/api/v1/admin/videos/BV1DEL000001").status_code == 404
    # 保留的场次不受影响
    assert db.get_video("BV1KEEP00001", 1) is not None

    # 3. 增量同步将本地删除同步到云端（零停服移除云端误导入场次与 FTS5 索引）
    del_delta = tmp_path / "del_delta.json.gz"
    exp = export_delta(db, build_manifest(cloud_db), del_delta)
    assert exp["parts_deleted"] == 2
    ap = apply_delta(cloud_db, del_delta)
    assert ap["parts_deleted"] == 2
    assert ap["subtitles_deleted"] == 1
    assert cloud_db.get_video("BV1DEL000001", 1) is None
    assert cloud_db.get_video("BV1DEL000001", 2) is None
    assert cloud_db.get_video("BV1KEEP00001", 1) is not None
    assert SearchService(cloud_db).search(SearchQuery(q="待删除测试台词独一无二"))["pagination"]["total"] == 0
