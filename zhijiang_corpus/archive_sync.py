"""早期录播（飞书多维表格）与「二创计画」合集补全 + 场景标签校准模块.

核心职责：
  1. 多 P 智能筛选 (`select_clean_pages`)：
     - 剔除末尾防撞车分 P（如「防撞」「A-SOUL防撞」「❤」或 <=15s 占位分 P）；
     - 同一 BV 下若同时存在无弹幕/纯净版与弹幕版，优先保留无弹幕版，不存在时再回退弹幕版；
     - 兼容未标注「弹幕」字样但前后两半时长与标题完全一致的成对双版本录播；
     - 兼容同时包含单曲切片与「完整录播」的早期视频（仅保留完整录播及回马枪/彩蛋）。
  2. 飞书早期录播补全（2020 ~ 2021-08-19 数据库起始边界前）：
     - 从缓存或飞书 clientvars JSON 解析全部「录播」条目；
     - 自动匹配数据库中已有的 2021-08-18/19 官方回放避免重复，补齐缺失早期场次。
  3. 二创计画（UID 547510303）10 个指定合集对齐：
     - 匹配数据库中已有场次，以二创计画合集分类为准修正并锁定错误 `scene_type`
       （保留 2025+ 真实生日会/周年 special 与明确多人联机突击）；
     - 过滤同场重复的「4K弹幕版/纯弹幕版」独立 BV 及「【直播剪辑】」；
     - 为数据库中缺失的场次（超管掐断/未生成官方回放/突击等）抓取分 P 字幕并增量入库。
"""

from __future__ import annotations

import base64
import gzip
import json
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .bilibili_client import (
    BilibiliAPIError,
    BilibiliClient,
    DEFAULT_LAN_PRIORITY,
    VideoInfo,
    VideoPageInfo,
)
from .cleaner import clean_bcc_payload
from .config import AppConfig
from .db import CorpusDB, SubtitleRow, VideoRecord

ProgressCallback = Callable[[str], None]

ERCHUANG_UID = 547510303

# (series_id, name, mapped_scene_type, solo_up_hint)
ERCHUANG_TARGET_SERIES: Tuple[Tuple[int, str, str, Optional[str]], ...] = (
    (3047489, "乃琳单播合集", "solo", "乃琳"),
    (3047492, "贝拉单播合集", "solo", "贝拉"),
    (3047498, "嘉然单播合集", "solo", "嘉然"),
    (3047495, "向晚单播合集", "solo", "向晚"),
    (3165958, "A-SOUL双播合集", "duo", None),
    (2105771, "A-SOUL夜谈", "group", None),
    (2105773, "A-SOUL游戏室", "group", None),
    (4498695, "团综/枝江节目", "group", None),
    (2105768, "A-SOUL小剧场", "group", None),
    (2105772, "◈特别企划！", "special", None),
)

SCENE_PRIORITY: Dict[str, int] = {
    "unknown": 0,
    "solo": 1,
    "duo": 2,
    "group": 3,
    "special": 4,
}

# 二创计画「◈特别企划！」最后更新日期（之后的新生日会仅归入单播合集，不应降级为 solo）
SPECIAL_SERIES_LAST_DATE = "2024-12-21"

ASOUL_MEMBERS = ("嘉然", "贝拉", "乃琳", "向晚", "珈乐")

_ANTI_COLLISION_RE = re.compile(
    r"(防撞|防撞车|占位|^[\s❤♥♡💗💖✨🎵🎶★☆\.。!！?？\-_~～]+$)",
    re.IGNORECASE,
)
_BONUS_PART_RE = re.compile(
    r"(回马枪|彩蛋|片头|片尾|花絮|开幕|闭幕|安可|加场|遗漏|\bOP\b|\bED\b|^[？?]+$)",
    re.IGNORECASE,
)
_DANMAKU_PART_RE = re.compile(r"弹幕")
_CLEAN_EXPLICIT_RE = re.compile(r"(无弹幕|纯净)")
_LEADING_P_RE = re.compile(r"^\s*[pP]\s*\d+\s*[-_.:：】\]\s]*")

_BOILERPLATE_BRACKETS_RE = re.compile(
    r"【(?:直播回放|直播录像|线上直播录像|完整录播|3D|2D|B限|A-SOUL|A-SOUL双播|"
    r"A-SOUL夜谈|A-SOUL游戏室|A-SOUL小剧场|A-SOUL团综|A-SOUL团播|枝江综艺|电台|"
    r"突击|突击直播|鸣潮/突击|突击/鸣潮|鸣潮突击|归环/突击|A-SOUL×[^】]*|A-SOULx[^】]*|"
    r"枝江娱乐|嘉然生日会|贝拉生日会|乃琳生日会|向晚生日会|A-SOUL[^】]*周年[^】]*)】",
    re.IGNORECASE,
)

_REAL_SPECIAL_TITLE_RE = re.compile(
    r"(生日会|生日派对|生日演唱会|周年纪念|周年直播|周年特别|五周年|四周年|三周年|二周年|一周年)"
)
_HOULITAN_RE = re.compile(r"(后日谈|预热|彩排|复盘)")

_FEISHU_SPECIAL_RE = re.compile(
    r"(生日会|生日派对|周年|纪念回|纪念直播|首播|BW20\d{2}|BW特别|萤火虫|特别企划|"
    r"合作直播|演唱会|音乐节|晚会|欧莱雅|花亦山|奇妙日|跨年|嘉年华|见面会|水友赛|"
    r"万粉|粉丝纪念|50万|100万|百万粉)"
)
_FEISHU_GROUP_RE = re.compile(
    r"(小剧场|夜谈|游戏室|团播|团综|全员|测试直播|运动会|狼人杀|谁是卧底|大乱斗)"
)
_FEISHU_DUO_RE = re.compile(r"(双播|双人|双排|二人)")

_DANMAKU_TITLE_TWIN_RE = re.compile(r"[（(][^）)]*弹幕[^）)]*[）)]")


@dataclass
class PageSelectionResult:
    selected: List[VideoPageInfo]
    dropped: List[Tuple[VideoPageInfo, str]]
    mode: str  # clean | danmaku_fallback | single | clean_half_dedup


@dataclass
class ArchiveVideoTask:
    bvid: str
    source: str  # feishu_early | erchuang_missing
    up_name: str
    scene_type: str
    stream_date: str
    pub_date_override: Optional[str]
    title_hint: str
    series_names: List[str] = field(default_factory=list)
    scene_reason: str = ""
    scene_locked: int = 1


@dataclass
class TagUpdatePlan:
    bvid: str
    up_name: str
    stream_date: str
    title: str
    old_scene_type: str
    new_scene_type: str
    old_locked: int
    series_names: List[str]
    matched_ec_bvids: List[str]


def is_anti_collision_page(p: VideoPageInfo, total_pages: int) -> bool:
    """判断单个分 P 是否为末尾防撞车分 P."""
    title = (p.part_title or "").strip()
    if _ANTI_COLLISION_RE.search(title):
        # 避免误伤明确为彩蛋/回马枪/问号花絮且时长较长的分 P（如 176s '？'）
        if p.duration_sec <= 60 or "防撞" in title or "占位" in title:
            return True
    if total_pages >= 2 and p.duration_sec <= 15 and not _BONUS_PART_RE.search(title):
        return True
    return False


def is_danmaku_page(p: VideoPageInfo) -> bool:
    title = (p.part_title or "").strip()
    if _CLEAN_EXPLICIT_RE.search(title):
        return False
    return bool(_DANMAKU_PART_RE.search(title))


def is_bonus_page(p: VideoPageInfo) -> bool:
    title = (p.part_title or "").strip()
    return bool(_BONUS_PART_RE.search(title))


def _strip_part_index(title: str) -> str:
    s = _LEADING_P_RE.sub("", (title or "").strip())
    return re.sub(r"\s+", "", s).lower()


def select_clean_pages(pages: Sequence[VideoPageInfo]) -> PageSelectionResult:
    """针对单部多 P 录播筛选应入库的分 P（去防撞 P -> 去冗余切片 -> 优先无弹幕 -> 去未标注双份）."""
    if not pages:
        return PageSelectionResult(selected=[], dropped=[], mode="empty")

    remaining = list(pages)
    dropped: List[Tuple[VideoPageInfo, str]] = []

    # 1) 剔除末尾防撞车分 P（仅从末尾剥离，保护前中段正常内容）
    while len(remaining) >= 2 and is_anti_collision_page(remaining[-1], len(remaining)):
        p = remaining.pop()
        dropped.append((p, "anti_collision_last_p"))

    if len(remaining) == 1:
        return PageSelectionResult(selected=remaining, dropped=dropped, mode="single")

    # 2) 若同时存在「完整录播」主分 P 与大量零散单曲切片（如 珈乐首播 P1~P12 歌曲 + P13~P14 完整录播），剔除单曲切片
    full_pages = [p for p in remaining if "完整录播" in (p.part_title or "")]
    if full_pages:
        full_non_dm_dur = sum(p.duration_sec for p in full_pages if not is_danmaku_page(p))
        full_total_dur = sum(p.duration_sec for p in full_pages)
        if max(full_non_dm_dur, full_total_dur) >= 1800:
            kept_step2: List[VideoPageInfo] = []
            for p in remaining:
                if "完整录播" in (p.part_title or "") or is_bonus_page(p) or p.duration_sec >= 1200:
                    kept_step2.append(p)
                else:
                    dropped.append((p, "redundant_song_clip_with_full_replay"))
            remaining = kept_step2

    # 3) 区分弹幕版与无弹幕/纯净版
    dm_pages = [p for p in remaining if is_danmaku_page(p)]
    clean_pages = [p for p in remaining if not is_danmaku_page(p)]

    if dm_pages and clean_pages:
        clean_main_dur = sum(p.duration_sec for p in clean_pages if not is_bonus_page(p))
        dm_main_dur = sum(p.duration_sec for p in dm_pages if not is_bonus_page(p))
        has_explicit_clean = any(_CLEAN_EXPLICIT_RE.search(p.part_title or "") for p in clean_pages)
        # 只要存在明确标明「纯净/无弹幕」的主分 P，或无弹幕主体时长达到弹幕主体 45% 以上
        # （兼容二创计画弹幕区同时放 P0 合并版 + P1..Pn 分段版导致弹幕总时长翻倍的情况），优先采用无弹幕版
        if (has_explicit_clean and clean_main_dur > 0) or (
            clean_main_dur > 0 and clean_main_dur >= int(dm_main_dur * 0.45)
        ) or (dm_main_dur == 0 and clean_pages):
            for p in dm_pages:
                dropped.append((p, "danmaku_version_skipped"))
            return PageSelectionResult(selected=clean_pages, dropped=dropped, mode="clean")

        # 否则说明主体只有弹幕版，clean_pages 只是未标弹幕的 OP/开幕/彩蛋等附带分 P：保留弹幕主体 + 不重复的附带分 P
        dm_durations = [p.duration_sec for p in dm_pages]
        kept_bonus: List[VideoPageInfo] = []
        for p in clean_pages:
            # 防止某个附带分 P 已有同长度的弹幕版重复
            if any(abs(p.duration_sec - d) <= 10 for d in dm_durations):
                dropped.append((p, "duplicate_of_danmaku_bonus"))
            else:
                kept_bonus.append(p)
        merged = sorted(dm_pages + kept_bonus, key=lambda x: x.page)
        return PageSelectionResult(selected=merged, dropped=dropped, mode="danmaku_fallback")

    if dm_pages and not clean_pages:
        return PageSelectionResult(selected=dm_pages, dropped=dropped, mode="danmaku_fallback")

    # 4) 无「弹幕」字样标注时，检测前后两半是否为时长与标签完全成对的双版本（如 贾布加布 6.3/6.4 漏标弹幕）
    if len(remaining) >= 2 and len(remaining) % 2 == 0:
        half = len(remaining) // 2
        first_half = remaining[:half]
        second_half = remaining[half:]
        all_paired = True
        for a, b in zip(first_half, second_half):
            max_d = max(a.duration_sec, b.duration_sec, 1)
            dur_close = abs(a.duration_sec - b.duration_sec) <= max(15, int(max_d * 0.02))
            label_same = _strip_part_index(a.part_title) == _strip_part_index(b.part_title)
            if not (dur_close and label_same):
                all_paired = False
                break
        if all_paired and sum(p.duration_sec for p in first_half) >= 600:
            for p in second_half:
                dropped.append((p, "unlabeled_duplicate_second_half"))
            return PageSelectionResult(
                selected=first_half, dropped=dropped, mode="clean_half_dedup"
            )

    return PageSelectionResult(selected=remaining, dropped=dropped, mode="clean")


def extract_stream_date(title: str, fallback_year: Optional[int] = None) -> Optional[str]:
    """从视频标题提取实际直播日期 YYYY-MM-DD."""
    if not title:
        return None
    # 完整 4 位年份：2024年05月12日 / 2024.05.12 / 2023.05..20 / 2021/3/10
    m = re.search(r"(20\d{2})\s*[年./\-]+\s*(\d{1,2})\s*[月./\-]+\s*(\d{1,2})", title)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    # 2 位年份：24.06.13
    m = re.search(r"(?<!\d)(2[0-6])\.(\d{1,2})\.(\d{1,2})(?!\d)", title)
    if m:
        y, mo, d = 2000 + int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    # 8 位紧凑日期：20210818
    m = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", title)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    # 仅月.日（早期飞书录播常见，如「7.15」「12.27」），需配合发布年份推断
    if fallback_year is not None:
        m = re.search(r"(?<![0-9.])(\d{1,2})[./月](\d{1,2})日?(?![0-9.])", title)
        if m:
            mo, d = int(m.group(1)), int(m.group(2))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                return f"{fallback_year:04d}-{mo:02d}-{d:02d}"
    return None


def infer_early_stream_date(title: str, pub_date: str) -> str:
    """结合标题月日与发布日期推断早期录播的真实直播日期（处理跨年如 12.31 发布于次年 1.1）."""
    explicit = extract_stream_date(title)
    if explicit:
        return explicit
    try:
        pub_dt = datetime.strptime(pub_date, "%Y-%m-%d")
    except Exception:
        return pub_date
    cand = extract_stream_date(title, fallback_year=pub_dt.year)
    if not cand:
        return pub_date
    # 若发布于 1 月而标题写 12 月，应为上一年
    try:
        cand_dt = datetime.strptime(cand, "%Y-%m-%d")
        if (cand_dt - pub_dt).days > 180:
            cand_dt = cand_dt.replace(year=pub_dt.year - 1)
            return cand_dt.strftime("%Y-%m-%d")
        if (pub_dt - cand_dt).days > 180:
            cand_dt = cand_dt.replace(year=pub_dt.year + 1)
            return cand_dt.strftime("%Y-%m-%d")
    except ValueError:
        return pub_date
    return cand


def normalize_title_for_match(title: str) -> str:
    s = _BOILERPLATE_BRACKETS_RE.sub(" ", title or "")
    s = _DANMAKU_TITLE_TWIN_RE.sub(" ", s)
    s = re.sub(r"20\d{2}\s*[年./\-]+\s*\d{1,2}\s*[月./\-]+\s*\d{1,2}日?(\d{1,2}点场)?", " ", s)
    s = re.sub(r"(?<!\d)2[0-6]\.\d{1,2}\.\d{1,2}(?!\d)", " ", s)
    s = re.sub(r"(?<![0-9.])\d{1,2}\.\d{1,2}(?![0-9.])", " ", s)
    s = re.sub(r"第?[一二三四五六七八九十百千万0-9]+期", " ", s)
    s = re.sub(
        r"(嘉然|贝拉|乃琳|向晚|心宜|思诺|珈乐|A-SOUL|ASOUL|直播录像|直播回放|生日会|纪念直播|突击直播)",
        " ",
        s,
        flags=re.IGNORECASE,
    )
    return re.sub(r"[\W_]+", "", s).lower()


def title_similarity(a: str, b: str) -> float:
    na, nb = normalize_title_for_match(a), normalize_title_for_match(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return (min(len(na), len(nb)) / max(len(na), len(nb))) * 0.95
    sa, sb = set(na), set(nb)
    return len(sa & sb) / max(len(sa), len(sb))


def _date_diff_days(d1: str, d2: str) -> int:
    try:
        dt1 = datetime.strptime(d1, "%Y-%m-%d")
        dt2 = datetime.strptime(d2, "%Y-%m-%d")
        return abs((dt1 - dt2).days)
    except Exception:
        return 999


def classify_feishu_early_video(title: str, pub_date: str) -> Tuple[str, str, str]:
    """根据飞书早期录播标题推断 (up_name, scene_type, stream_date)."""
    sdate = infer_early_stream_date(title, pub_date)
    found_members = [m for m in ASOUL_MEMBERS if m in title]
    is_special = bool(_FEISHU_SPECIAL_RE.search(title)) and not bool(_HOULITAN_RE.search(title))
    is_group = bool(_FEISHU_GROUP_RE.search(title))
    is_duo = bool(_FEISHU_DUO_RE.search(title))

    if is_special:
        scene_type = "special"
        up_name = found_members[0] if len(found_members) == 1 else "A-SOUL"
    elif is_group or len(found_members) >= 3:
        scene_type = "group"
        up_name = "A-SOUL"
    elif is_duo or len(found_members) == 2:
        scene_type = "duo"
        up_name = "A-SOUL"
    elif len(found_members) == 1:
        scene_type = "solo"
        up_name = found_members[0]
    else:
        scene_type = "group"
        up_name = "A-SOUL"
    return up_name, scene_type, sdate


def load_feishu_replays_from_clientvars(clientvars_path: Path) -> List[Dict[str, Any]]:
    """从飞书多维表格 clientvars 原始 JSON 提取全部「录播」条目."""
    raw = json.loads(clientvars_path.read_text(encoding="utf-8"))
    table = json.loads(gzip.decompress(base64.b64decode(raw["data"]["table"])).decode("utf-8"))
    field_map = table["fieldMap"]
    type_opts = {o["id"]: o["name"] for o in field_map["fld4ufRWRi"]["property"]["options"]}

    out: List[Dict[str, Any]] = []
    for rid, rec in table["recordMap"].items():
        t_cell = rec.get("fld4ufRWRi")
        t_val = t_cell.get("value") if isinstance(t_cell, dict) else t_cell
        if type_opts.get(t_val) != "录播":
            continue
        title_cell = rec.get("fldYrrV1Og")
        title_val = title_cell.get("value") if isinstance(title_cell, dict) else title_cell
        text, link = "", ""
        if isinstance(title_val, list):
            for it in title_val:
                if isinstance(it, dict):
                    if it.get("text"):
                        text += str(it["text"])
                    if it.get("link"):
                        link = str(it["link"])
        m = re.search(r"(BV[0-9A-Za-z]{10})", link or text)
        if not m:
            continue
        out.append({"rid": rid, "bvid": m.group(1), "text": text.strip(), "link": link.strip()})
    return out


def _load_db_sessions(db: CorpusDB) -> Dict[str, Dict[str, Any]]:
    with db.session() as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                """
                SELECT bvid, page, up_name, raw_up_name, title, pub_date,
                       duration_sec, scene_type, scene_locked, subtitle_status
                  FROM videos
                """
            ).fetchall()
        ]
    by_bv: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_bv[str(r["bvid"])].append(r)

    sessions: Dict[str, Dict[str, Any]] = {}
    for bvid, rlist in by_bv.items():
        r0 = rlist[0]
        title = str(r0["title"] or "")
        pub_date = str(r0["pub_date"] or "")
        sdate = extract_stream_date(title) or pub_date
        sessions[bvid] = {
            "bvid": bvid,
            "up_name": str(r0["up_name"] or ""),
            "pub_date": pub_date,
            "stream_date": sdate,
            "scene_type": str(r0["scene_type"] or "unknown"),
            "scene_locked": int(r0["scene_locked"] or 0),
            "title": title,
            "tuji": "突击" in title,
            "pages": len(rlist),
            "total_dur": sum(int(x["duration_sec"] or 0) for x in rlist),
        }
    return sessions


def _should_preserve_existing_scene(db_sess: Dict[str, Any], ec_scene: str) -> Optional[str]:
    """针对二创计画合集边界特例的保护规则（返回保护原因，None 表示按二创计画修改）."""
    old_sc = db_sess["scene_type"]
    title = db_sess["title"]
    sdate = db_sess["stream_date"]

    # 1) 2025+ 的真实生日会/周年纪念：因「◈特别企划！」停更于 2024-12-21，二创计画仅放在单播合集里
    if (
        old_sc == "special"
        and ec_scene == "solo"
        and sdate > SPECIAL_SERIES_LAST_DATE
        and _REAL_SPECIAL_TITLE_RE.search(title)
        and not _HOULITAN_RE.search(title)
    ):
        return "preserve_post_2024_birthday_or_anniversary_special"

    # 2) 明确标题写着「突击」且「和队友一起」的多人联机视角（二创计画按个人视角放入单播合集）
    if old_sc in ("duo", "group") and ec_scene == "solo" and "突击" in title:
        t_no_self = title.replace(db_sess["up_name"], "")
        if re.search(
            r"和[^，。！!？?]*?(然比|然然|嘉然|拉姐|拉拉|贝拉|乃宝|乃琳|小琳|向晚|晚晚|汐汐|高人)[^，。！!？?]*?一起",
            t_no_self,
        ):
            return "preserve_multi_member_tuji_collab"

    return None


def plan_erchuang_sync(
    db: CorpusDB,
    series_data: Dict[str, Any],
) -> Tuple[List[TagUpdatePlan], List[ArchiveVideoTask], Dict[str, Any]]:
    """比对「二创计画」10 个合集与现有数据库，生成标签修正清单与缺失场次补录清单."""
    db_sessions = _load_db_sessions(db)
    db_by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in db_sessions.values():
        db_by_date[d["stream_date"]].append(d)
        if d["pub_date"] != d["stream_date"]:
            db_by_date[d["pub_date"]].append(d)

    series_meta = {name: (sc, up_hint) for _, name, sc, up_hint in ERCHUANG_TARGET_SERIES}

    # 合并多合集归属 + 跳过【直播剪辑】
    ec_map: Dict[str, Dict[str, Any]] = {}
    skipped_clips = 0
    for info in series_data.values():
        sname = str(info["name"])
        if sname not in series_meta:
            continue
        sc, up_hint = series_meta[sname]
        for v in info.get("videos") or []:
            title = str(v.get("title") or "")
            if "【直播剪辑】" in title:
                skipped_clips += 1
                continue
            bv = str(v.get("bvid") or "").strip()
            if not bv:
                continue
            if bv not in ec_map:
                pub_date = str(v.get("pub_date") or "")
                sdate = extract_stream_date(title) or pub_date
                ec_map[bv] = {
                    "bvid": bv,
                    "title": title,
                    "pub_date": pub_date,
                    "stream_date": sdate,
                    "duration_sec": int(v.get("duration_sec") or 0),
                    "series": [sname],
                    "scene_type": sc,
                    "up_hint": up_hint,
                    "tuji": "突击" in title,
                    "members": [m for m in ASOUL_MEMBERS if m in title],
                    "is_danmaku_twin": bool(_DANMAKU_TITLE_TWIN_RE.search(title)),
                }
            else:
                if sname not in ec_map[bv]["series"]:
                    ec_map[bv]["series"].append(sname)
                if SCENE_PRIORITY[sc] > SCENE_PRIORITY[ec_map[bv]["scene_type"]]:
                    ec_map[bv]["scene_type"] = sc
                if up_hint and not ec_map[bv]["up_hint"]:
                    ec_map[bv]["up_hint"] = up_hint

    # 同一天若存在同标题的「(4K弹幕版)/(纯弹幕版)」独立 BV 与无弹幕 BV，优先保留无弹幕 BV
    by_sd: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ev in ec_map.values():
        by_sd[ev["stream_date"]].append(ev)

    active_ec: List[Dict[str, Any]] = []
    dropped_danmaku_twins = 0
    for sd, items in by_sd.items():
        for ev in items:
            if ev["is_danmaku_twin"]:
                twin = [
                    x
                    for x in items
                    if x["bvid"] != ev["bvid"]
                    and not x["is_danmaku_twin"]
                    and title_similarity(ev["title"], x["title"]) >= 0.85
                ]
                if twin:
                    dropped_danmaku_twins += 1
                    continue
            active_ec.append(ev)

    # 两轮精准匹配：
    #   Pass 0: 同 BV 已在库中（幂等重跑时直接命中）
    #   Pass 1: 日期一致+标题相似>=0.35，或日期差<=3天+标题相似>=0.65
    #   Pass 2: 同日剩余未匹配项（含多场合集 '+'、同日突击、同日单场）
    matched_ec_to_db: Dict[str, List[str]] = {}
    used_db: set[str] = set()

    for ev in active_ec:
        if ev["bvid"] in db_sessions:
            matched_ec_to_db[ev["bvid"]] = [ev["bvid"]]
            used_db.add(ev["bvid"])

    for ev in active_ec:
        if ev["bvid"] in matched_ec_to_db:
            continue
        sd = ev["stream_date"]
        try:
            dt = datetime.strptime(sd, "%Y-%m-%d")
        except Exception:
            continue
        cands: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for delta in range(-3, 4):
            ds = (dt + timedelta(days=delta)).strftime("%Y-%m-%d")
            for c in db_by_date.get(ds, []):
                if c["bvid"] not in seen and c["up_name"] not in ("心宜", "思诺"):
                    seen.add(c["bvid"])
                    cands.append(c)

        best: Optional[Dict[str, Any]] = None
        best_score = -1.0
        for c in cands:
            if ev["up_hint"] and ev["scene_type"] == "solo" and c["up_name"] != ev["up_hint"]:
                continue
            dd = min(_date_diff_days(sd, c["stream_date"]), _date_diff_days(sd, c["pub_date"]))
            sim = title_similarity(ev["title"], c["title"])
            same_sd = sd == c["stream_date"]
            if not ((same_sd and sim >= 0.35) or (dd <= 3 and sim >= 0.65)):
                continue
            score = sim * 10.0 + (4.0 if same_sd else max(0.0, 2.5 - dd * 0.7))
            if ev["up_hint"] and c["up_name"] == ev["up_hint"]:
                score += 2.0
            if ev["tuji"] == c["tuji"]:
                score += 1.0
            if score > best_score:
                best_score = score
                best = c
        if best is not None:
            matched_ec_to_db[ev["bvid"]] = [best["bvid"]]
            used_db.add(best["bvid"])

    for ev in active_ec:
        if ev["bvid"] in matched_ec_to_db:
            continue
        sd = ev["stream_date"]
        cands = [
            c
            for c in db_by_date.get(sd, [])
            if c["bvid"] not in used_db
            and c["stream_date"] == sd
            and c["up_name"] not in ("心宜", "思诺")
        ]
        if ev["up_hint"] and ev["scene_type"] == "solo":
            cands = [c for c in cands if c["up_name"] == ev["up_hint"]]
            ev_norm = normalize_title_for_match(ev["title"])
            sub_hits = [
                c
                for c in cands
                if title_similarity(ev["title"], c["title"]) >= 0.25
                or (
                    normalize_title_for_match(c["title"])
                    and normalize_title_for_match(c["title"]) in ev_norm
                )
            ]
            if sub_hits:
                matched_ec_to_db[ev["bvid"]] = [c["bvid"] for c in sub_hits]
                for c in sub_hits:
                    used_db.add(c["bvid"])
                continue
            same_t = [c for c in cands if c["tuji"] == ev["tuji"]]
            if len(same_t) == 1:
                matched_ec_to_db[ev["bvid"]] = [same_t[0]["bvid"]]
                used_db.add(same_t[0]["bvid"])
                continue
            if len(cands) == 1:
                matched_ec_to_db[ev["bvid"]] = [cands[0]["bvid"]]
                used_db.add(cands[0]["bvid"])
                continue
        else:
            sub_hits = [c for c in cands if title_similarity(ev["title"], c["title"]) >= 0.22]
            if len(sub_hits) == 1:
                matched_ec_to_db[ev["bvid"]] = [sub_hits[0]["bvid"]]
                used_db.add(sub_hits[0]["bvid"])
                continue
            if ev["members"]:
                m_cands = [c for c in cands if c["up_name"] in ev["members"] and not c["tuji"]]
                if len(m_cands) == 1:
                    matched_ec_to_db[ev["bvid"]] = [m_cands[0]["bvid"]]
                    used_db.add(m_cands[0]["bvid"])
                    continue

    # 生成标签修正清单
    db_to_ecs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ebv, dbvs in matched_ec_to_db.items():
        for dbv in dbvs:
            db_to_ecs[dbv].append(ec_map[ebv])

    tag_updates: List[TagUpdatePlan] = []
    preserved_tags: List[Dict[str, Any]] = []
    for dbv, evs in sorted(db_to_ecs.items()):
        d = db_sessions[dbv]
        target_sc = max((e["scene_type"] for e in evs), key=lambda x: SCENE_PRIORITY[x])
        if d["scene_type"] == target_sc:
            continue
        guard = _should_preserve_existing_scene(d, target_sc)
        if guard:
            preserved_tags.append(
                {
                    "bvid": dbv,
                    "up_name": d["up_name"],
                    "stream_date": d["stream_date"],
                    "title": d["title"],
                    "kept_scene": d["scene_type"],
                    "ec_scene": target_sc,
                    "guard": guard,
                }
            )
            continue
        series_all = sorted({sn for e in evs for sn in e["series"]})
        tag_updates.append(
            TagUpdatePlan(
                bvid=dbv,
                up_name=d["up_name"],
                stream_date=d["stream_date"],
                title=d["title"],
                old_scene_type=d["scene_type"],
                new_scene_type=target_sc,
                old_locked=d["scene_locked"],
                series_names=series_all,
                matched_ec_bvids=[e["bvid"] for e in evs],
            )
        )

    # 生成缺失场次补录清单
    missing_tasks: List[ArchiveVideoTask] = []
    for ev in sorted(active_ec, key=lambda x: (x["stream_date"], x["bvid"])):
        if ev["bvid"] in matched_ec_to_db:
            continue
        # 确定 up_name：单播或单人特别场次归属该成员，多人/双播/团综/团体企划归属 A-SOUL
        if ev["up_hint"]:
            up_name = ev["up_hint"]
        elif ev["scene_type"] == "special" and len(ev["members"]) == 1:
            up_name = ev["members"][0]
        else:
            up_name = "A-SOUL"
        # 2025+ 生日会单独归在单播合集的情况，入库时直接标 special
        eff_scene = ev["scene_type"]
        if (
            eff_scene == "solo"
            and ev["stream_date"] > SPECIAL_SERIES_LAST_DATE
            and _REAL_SPECIAL_TITLE_RE.search(ev["title"])
            and not _HOULITAN_RE.search(ev["title"])
        ):
            eff_scene = "special"

        pub_override = (
            ev["stream_date"]
            if _date_diff_days(ev["stream_date"], ev["pub_date"]) > 5
            else None
        )
        missing_tasks.append(
            ArchiveVideoTask(
                bvid=ev["bvid"],
                source="erchuang_missing",
                up_name=up_name,
                scene_type=eff_scene,
                stream_date=ev["stream_date"],
                pub_date_override=pub_override,
                title_hint=ev["title"],
                series_names=list(ev["series"]),
                scene_reason=json.dumps(
                    {"source": "erchuang_series", "series": ev["series"]},
                    ensure_ascii=False,
                ),
                scene_locked=1,
            )
        )

    stats = {
        "ec_unique_bvids": len(ec_map),
        "skipped_clips": skipped_clips,
        "dropped_danmaku_twins": dropped_danmaku_twins,
        "active_ec": len(active_ec),
        "matched_ec": len(matched_ec_to_db),
        "matched_db": len(db_to_ecs),
        "tag_updates": len(tag_updates),
        "preserved_tags": preserved_tags,
        "missing_to_ingest": len(missing_tasks),
    }
    return tag_updates, missing_tasks, stats


def plan_feishu_early_sync(
    db: CorpusDB,
    feishu_infos: Dict[str, Dict[str, Any]],
    *,
    cutoff_date: str = "2021-08-19",
) -> Tuple[List[ArchiveVideoTask], Dict[str, Any]]:
    """基于飞书早期录播详情缓存，筛选 <= cutoff_date 且数据库中尚未收录的早期场次."""
    db_sessions = _load_db_sessions(db)
    db_by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in db_sessions.values():
        db_by_date[d["stream_date"]].append(d)
        if d["pub_date"] != d["stream_date"]:
            db_by_date[d["pub_date"]].append(d)

    tasks: List[ArchiveVideoTask] = []
    skipped_after_cutoff = 0
    skipped_already_in_db = 0
    skipped_invalid = 0

    for bv, info in sorted(
        feishu_infos.items(), key=lambda kv: (kv[1].get("pub_date", "9999"), kv[0])
    ):
        if not info.get("ok"):
            skipped_invalid += 1
            continue
        title = str(info.get("title") or info.get("feishu_text") or "")
        pub_date = str(info.get("pub_date") or "")
        up_name, scene_type, sdate = classify_feishu_early_video(title, pub_date)

        # 截断到数据库起始日期边界（含边界日缺失场次，如 2021-08-18）
        if sdate > cutoff_date:
            skipped_after_cutoff += 1
            continue

        # 已同 BV 入库，或同日已有匹配官方回放 -> 跳过
        if bv in db_sessions:
            skipped_already_in_db += 1
            continue

        already_covered = False
        try:
            dt = datetime.strptime(sdate, "%Y-%m-%d")
        except Exception:
            dt = None
        if dt is not None:
            for delta in (-1, 0, 1):
                ds = (dt + timedelta(days=delta)).strftime("%Y-%m-%d")
                for c in db_by_date.get(ds, []):
                    if scene_type == "solo" and up_name in ASOUL_MEMBERS and c["up_name"] != up_name:
                        continue
                    if title_similarity(title, c["title"]) >= 0.45:
                        already_covered = True
                        break
                    if (
                        c["stream_date"] == sdate
                        and up_name in ASOUL_MEMBERS
                        and c["up_name"] == up_name
                    ):
                        already_covered = True
                        break
                if already_covered:
                    break
        if already_covered:
            skipped_already_in_db += 1
            continue

        pub_override = sdate if _date_diff_days(sdate, pub_date) > 5 else None
        tasks.append(
            ArchiveVideoTask(
                bvid=bv,
                source="feishu_early",
                up_name=up_name,
                scene_type=scene_type,
                stream_date=sdate,
                pub_date_override=pub_override,
                title_hint=title,
                series_names=["飞书早期录播"],
                scene_reason=json.dumps(
                    {"source": "feishu_early", "inferred_from_title": True},
                    ensure_ascii=False,
                ),
                scene_locked=0,
            )
        )

    stats = {
        "feishu_cached_total": len(feishu_infos),
        "skipped_invalid": skipped_invalid,
        "skipped_after_cutoff": skipped_after_cutoff,
        "skipped_already_in_db": skipped_already_in_db,
        "early_to_ingest": len(tasks),
        "date_min": tasks[0].stream_date if tasks else None,
        "date_max": tasks[-1].stream_date if tasks else None,
    }
    return tasks, stats


def apply_tag_updates(
    db: CorpusDB,
    plans: Sequence[TagUpdatePlan],
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """按二创计画分类修正数据库中已有场次的 scene_type 并锁定."""
    updated_videos = 0
    updated_rows = 0
    if not dry_run:
        for p in plans:
            reason = json.dumps(
                {
                    "source": "erchuang_series",
                    "series": p.series_names,
                    "ec_bvids": p.matched_ec_bvids,
                    "old_scene_type": p.old_scene_type,
                },
                ensure_ascii=False,
            )
            n = db.update_scene_type(
                p.bvid,
                None,
                p.new_scene_type,
                confidence=1.0,
                reason=reason,
                lock=True,
            )
            if n > 0:
                updated_videos += 1
                updated_rows += n
    return {
        "planned_videos": len(plans),
        "updated_videos": updated_videos if not dry_run else 0,
        "updated_video_rows": updated_rows if not dry_run else 0,
        "dry_run": dry_run,
    }


def _video_info_from_cache(raw: Dict[str, Any]) -> VideoInfo:
    pages = [
        VideoPageInfo(
            page=int(p["page"]),
            cid=int(p["cid"]),
            part_title=str(p.get("part_title") or ""),
            duration_sec=int(p.get("duration_sec") or 0),
        )
        for p in raw.get("pages") or []
    ]
    return VideoInfo(
        bvid=str(raw["bvid"]),
        aid=int(raw.get("aid") or 0),
        title=str(raw.get("title") or ""),
        desc=str(raw.get("desc") or ""),
        up_uid=int(raw.get("up_uid") or 0),
        raw_up_name=str(raw.get("raw_up_name") or ""),
        pub_ts=int(raw.get("pub_ts") or 0),
        pub_date=str(raw.get("pub_date") or ""),
        duration_sec=int(raw.get("duration_sec") or 0),
        pages=pages,
    )


def ingest_archive_tasks(
    db: CorpusDB,
    config: AppConfig,
    tasks: Sequence[ArchiveVideoTask],
    *,
    video_info_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    prefer_lans: Sequence[str] = DEFAULT_LAN_PRIORITY,
    force: bool = False,
    retry_no_sub: bool = False,
    cooldown_every: int = 30,
    cooldown_sec: float = 6.0,
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """对过滤后的外部录播任务执行多 P 清洗筛选、字幕抓取与增量入库."""
    db.init_db()
    run_id = uuid.uuid4().hex[:12]

    def emit(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    stats: Dict[str, Any] = {
        "run_id": run_id,
        "tasks_total": len(tasks),
        "videos_fetched": 0,
        "parts_selected": 0,
        "parts_dropped": 0,
        "parts_ok": 0,
        "parts_skipped": 0,
        "parts_no_subtitle": 0,
        "parts_failed": 0,
        "lines_inserted": 0,
        "lines_duplicate": 0,
        "errors": [],
    }

    cache = video_info_cache or {}
    network_videos = 0

    with BilibiliClient(
        cookie=config.bilibili_cookie,
        delay=config.request_delay,
        jitter=config.request_jitter,
        max_retries=config.max_retries,
    ) as bili:
        for idx, task in enumerate(tasks, start=1):
            bvid = task.bvid.strip()
            if not bvid:
                continue

            # 若整部视频已在库且非 force，快速跳过
            if not force and db.is_bvid_completed(bvid, allow_no_subtitle=not retry_no_sub):
                stats["parts_skipped"] += 1
                emit(f"[{idx}/{len(tasks)}] 已完整入库，跳过: {bvid}")
                continue

            if cooldown_every > 0 and network_videos > 0 and network_videos % cooldown_every == 0:
                emit(f"[限速冷却] 已连续处理 {network_videos} 场，休息 {cooldown_sec:.0f}s...")
                time.sleep(cooldown_sec)

            cached_raw = cache.get(bvid)
            if cached_raw and cached_raw.get("ok") and cached_raw.get("pages"):
                vinfo = _video_info_from_cache(cached_raw)
            else:
                try:
                    vinfo = bili.get_video_info(bvid)
                except Exception as exc:
                    stats["parts_failed"] += 1
                    msg = f"[{idx}/{len(tasks)}] {bvid} 获取视频详情失败: {exc}"
                    stats["errors"].append(msg)
                    emit(msg)
                    db.log_ingest(run_id, bvid, 0, "failed", msg)
                    continue

            stats["videos_fetched"] += 1
            sel = select_clean_pages(vinfo.pages)
            stats["parts_selected"] += len(sel.selected)
            stats["parts_dropped"] += len(sel.dropped)

            eff_pub_date = task.pub_date_override or vinfo.pub_date
            extra_meta = {
                "aid": vinfo.aid,
                "desc_snippet": (vinfo.desc or "")[:500],
                "archive_source": task.source,
                "archive_series": task.series_names,
                "stream_date": task.stream_date,
                "page_select_mode": sel.mode,
                "selected_pages": [p.page for p in sel.selected],
                "dropped_pages": [
                    {"page": p.page, "part": p.part_title, "reason": r}
                    for p, r in sel.dropped
                ],
            }

            did_network = False
            for p in sel.selected:
                existing = db.get_video(vinfo.bvid, p.page)
                if existing and not force:
                    st = existing.get("subtitle_status")
                    if st == "ok" or (st in ("no_subtitle", "asr_empty") and not retry_no_sub):
                        stats["parts_skipped"] += 1
                        continue

                did_network = True
                try:
                    tracks = bili.list_subtitle_tracks(vinfo.bvid, p.cid)
                    chosen = bili.pick_subtitle_track(tracks, prefer_lans=prefer_lans)
                    if chosen is None:
                        vid_rec = VideoRecord(
                            bvid=vinfo.bvid,
                            page=p.page,
                            cid=p.cid,
                            up_uid=vinfo.up_uid,
                            up_name=task.up_name,
                            raw_up_name=vinfo.raw_up_name,
                            title=vinfo.title,
                            part_title=p.part_title,
                            pub_date=eff_pub_date,
                            pub_ts=vinfo.pub_ts,
                            duration_sec=p.duration_sec or vinfo.duration_sec,
                            subtitle_lan="",
                            subtitle_status="no_subtitle" if bili.has_sessdata else "pending",
                            line_count=0,
                            scene_type=task.scene_type,
                            scene_confidence=1.0 if task.scene_locked else 0.85,
                            scene_reason=task.scene_reason,
                            scene_locked=task.scene_locked,
                            extra_meta=extra_meta,
                        )
                        db.save_video_and_subtitles(vid_rec, [], replace_existing=False)
                        stats["parts_no_subtitle"] += 1
                        emit(
                            f"[{idx}/{len(tasks)}] 无字幕轨: [{task.up_name}/{task.scene_type}] "
                            f"{vinfo.bvid} P{p.page} ({eff_pub_date}) {vinfo.title[:32]}"
                        )
                        db.log_ingest(run_id, vinfo.bvid, p.page, "no_subtitle", "no subtitle track")
                        continue

                    raw_sub = bili.fetch_subtitle_json(chosen.subtitle_url)
                    cleaned_items = clean_bcc_payload(
                        raw_sub, up_name=task.up_name, pub_date=eff_pub_date
                    )
                    source_tag = (
                        "bilibili_ai"
                        if chosen.lan.startswith("ai-") or chosen.ai_type > 0
                        else "bilibili_cc"
                    )
                    rows = [
                        SubtitleRow(
                            up_name=task.up_name,
                            bvid=vinfo.bvid,
                            page=p.page,
                            pub_date=eff_pub_date,
                            start_sec=item.start_sec,
                            end_sec=item.end_sec,
                            content=item.content,
                            scene_type=task.scene_type,
                            is_visible=1,
                            source=source_tag,
                        )
                        for item in cleaned_items
                    ]
                    vid_rec = VideoRecord(
                        bvid=vinfo.bvid,
                        page=p.page,
                        cid=p.cid,
                        up_uid=vinfo.up_uid,
                        up_name=task.up_name,
                        raw_up_name=vinfo.raw_up_name,
                        title=vinfo.title,
                        part_title=p.part_title,
                        pub_date=eff_pub_date,
                        pub_ts=vinfo.pub_ts,
                        duration_sec=p.duration_sec or vinfo.duration_sec,
                        subtitle_lan=chosen.lan,
                        subtitle_status="ok" if rows else "no_subtitle",
                        line_count=len(rows),
                        scene_type=task.scene_type,
                        scene_confidence=1.0 if task.scene_locked else 0.85,
                        scene_reason=task.scene_reason,
                        scene_locked=task.scene_locked,
                        extra_meta=extra_meta,
                    )
                    inserted, dup = db.save_video_and_subtitles(
                        vid_rec, rows, replace_existing=force
                    )
                    stats["lines_inserted"] += inserted
                    stats["lines_duplicate"] += dup
                    if rows:
                        stats["parts_ok"] += 1
                        emit(
                            f"[{idx}/{len(tasks)}] 入库成功: [{task.up_name}/{task.scene_type}] "
                            f"{vinfo.bvid} P{p.page} ({eff_pub_date}) [{chosen.lan}] +{inserted}条 "
                            f"| {vinfo.title[:30]}"
                        )
                        db.log_ingest(
                            run_id,
                            vinfo.bvid,
                            p.page,
                            "ok",
                            f"source={task.source} lan={chosen.lan} inserted={inserted}",
                        )
                    else:
                        stats["parts_no_subtitle"] += 1
                        db.log_ingest(run_id, vinfo.bvid, p.page, "no_subtitle", "cleaned lines=0")
                except (BilibiliAPIError, Exception) as exc:
                    stats["parts_failed"] += 1
                    msg = f"[{idx}/{len(tasks)}] {vinfo.bvid} P{p.page} 抓取字幕失败: {exc}"
                    stats["errors"].append(msg)
                    emit(msg)
                    db.log_ingest(run_id, vinfo.bvid, p.page, "failed", msg)

            if did_network:
                network_videos += 1

    return stats
