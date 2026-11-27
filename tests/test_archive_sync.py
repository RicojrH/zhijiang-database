"""多 P 清洗筛选与外部录播对齐规则单元测试."""

from zhijiang_corpus.archive_sync import (
    classify_feishu_early_video,
    extract_stream_date,
    select_clean_pages,
)
from zhijiang_corpus.bilibili_client import VideoPageInfo


def _p(page: int, dur: int, title: str) -> VideoPageInfo:
    return VideoPageInfo(page=page, cid=1000 + page, part_title=title, duration_sec=dur)


def test_prefer_clean_and_drop_anti_collision():
    pages = [
        _p(1, 4801, "P1 弹幕版【4K】"),
        _p(2, 2603, "P2 弹幕版【4K】"),
        _p(3, 4803, "P3 纯净版"),
        _p(4, 2603, "P4 纯净版"),
        _p(5, 6, "A-SOUL防撞"),
    ]
    res = select_clean_pages(pages)
    assert res.mode == "clean"
    assert [p.page for p in res.selected] == [3, 4]
    assert {p.page for p, _ in res.dropped} == {1, 2, 5}


def test_prefer_explicit_clean_even_when_danmaku_has_p0_plus_splits():
    pages = [
        _p(1, 10501, "P0 弹幕版【4K】"),
        _p(2, 4952, "P1 弹幕版【4K】"),
        _p(3, 5547, "P2 弹幕版【4K】"),
        _p(4, 10498, "P3 纯净版"),
        _p(5, 6, "A-SOUL防撞"),
    ]
    res = select_clean_pages(pages)
    assert res.mode == "clean"
    assert [p.page for p in res.selected] == [4]


def test_fallback_to_danmaku_when_no_clean_exists():
    pages = [
        _p(1, 5116, "P1 弹幕版【4K】"),
        _p(2, 5395, "P2 弹幕版【4K】"),
        _p(3, 6, "防撞A-SOUL"),
    ]
    res = select_clean_pages(pages)
    assert res.mode == "danmaku_fallback"
    assert [p.page for p in res.selected] == [1, 2]
    assert [p.page for p, _ in res.dropped] == [3]


def test_unlabeled_duplicate_halves_dedup():
    pages = [
        _p(1, 3240, "p1 录播 上"),
        _p(2, 4589, "p2 录播 下"),
        _p(3, 3240, "p3 录播 上"),
        _p(4, 4589, "p4 录播 下"),
    ]
    res = select_clean_pages(pages)
    assert res.mode == "clean_half_dedup"
    assert [p.page for p in res.selected] == [1, 2]


def test_full_replay_drops_redundant_song_clips():
    pages = [
        _p(1, 300, "p1 《Snow halation》"),
        _p(2, 240, "p2 《我要你》"),
        _p(3, 3556, "p13 完整录播 上"),
        _p(4, 3925, "p14 完整录播 下"),
    ]
    res = select_clean_pages(pages)
    assert [p.page for p in res.selected] == [3, 4]


def test_keep_short_bonus_huimaqiang_and_drop_heart_collision():
    pages = [
        _p(1, 3617, "P1 录播 上"),
        _p(2, 3963, "P2 录播 下"),
        _p(3, 57, "P3 回马枪"),
        _p(4, 4, "❤"),
    ]
    res = select_clean_pages(pages)
    assert [p.page for p in res.selected] == [1, 2, 3]
    assert [p.page for p, _ in res.dropped] == [4]


def test_short_video_prefers_clean_over_danmaku():
    pages = [
        _p(1, 228, "P1 录播"),
        _p(2, 227, "P2 录播【弹幕】"),
        _p(3, 6, "❤v❤"),
    ]
    res = select_clean_pages(pages)
    assert res.mode == "clean"
    assert [p.page for p in res.selected] == [1]


def test_date_and_feishu_classification():
    assert extract_stream_date("【A-SOUL】乃琳 24.06.13 台词大挑战") == "2024-06-13"
    assert extract_stream_date("【A-SOUL】嘉然 2023.05..20 520直播") == "2023-05-20"
    assert classify_feishu_early_video("【A-SOUL】12.27 珈乐Carol首播【直播录像】", "2020-12-28") == (
        "珈乐",
        "special",
        "2020-12-27",
    )
    assert classify_feishu_early_video("【A-SOUL】乃琳/向晚 7.15 琴 瑟 和 鸣【直播录像】", "2021-07-16") == (
        "A-SOUL",
        "duo",
        "2021-07-15",
    )
    assert classify_feishu_early_video("【A-SOUL小剧场】第一期 12.20 小剧场来啦", "2020-12-21") == (
        "A-SOUL",
        "group",
        "2020-12-20",
    )
