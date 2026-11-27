"""轻量场景初筛引擎（单播 / 双播 / 团播 / 联动 / 特别场次）.

设计原则：
  1. 非破坏性：仅更新 scene_type / scene_confidence / scene_reason，绝不修改字幕正文；
  2. 尊重人工标注：scene_locked = 1 的场次默认跳过；
  3. 可解释：每条打标结果均输出命中规则摘要，支持 --dry-run 预览后再落库。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .config import AppConfig, MemberConfig
from .db import CorpusDB

# 标题 / 简介强特征词典
SPECIAL_TITLE_KEYWORDS = (
    "生日会",
    "生日直播",
    "周年",
    "纪念日",
    "嘉年华",
    "演唱会",
    "跨年晚会",
    "中秋晚会",
    "春节联欢",
    "特别晚会",
    "特别舞台",
    "毕业",
    "出道",
)

GROUP_TITLE_KEYWORDS = (
    "团播",
    "团综",
    "全员",
    "大乱斗",
    "小剧场",
    "枝江夜话",
    "游戏夜",
    "运动会",
    "狼人杀",
    "谁是卧底",
    "A-SOUL团",
    "枝江学园",
)

DUO_TITLE_KEYWORDS = (
    "双播",
    "双人",
    "双排",
    "双影",
    "小心思",
    "双人茶话会",
    "双人合唱",
    "二人",
)

COLLAB_TITLE_KEYWORDS = (
    "联动",
    "连麦",
    "嘉宾",
    "做客",
    "串门",
    "合作直播",
)

_TUJI_TAG_RE = re.compile(r"【[^】]*突击[^】]*】")
_TUJI_MULTI_HINTS = (
    "双播",
    "双人",
    "双排",
    "双影",
    "小心思",
    "二人",
    "团播",
    "团综",
    "全员",
    "联动",
    "连麦",
    "琳嘉",
    "一起玩游戏",
    "高人高人",
    "小黄诺和小粉宜",
    "亚当重锤",
)

# 字幕正文多人互动特征词（区分“单播顺嘴提一句队友”与“真的在现场互动”）
MULTI_SPEAKER_SUB_CUES = (
    "大家好我们是",
    "我们两个",
    "我们俩",
    "轮到你了",
    "下一位",
    "抽签",
    "分组",
    "裁判",
    "卧底",
    "双人合唱",
    "全员",
    "连麦",
)

# 强团播口播信号（仅靠此信号才可在无多成员名时疑似团播）
STRONG_GROUP_SUB_CUES = (
    "大家好我们是",
    "全员",
    "抽签",
    "分组",
    "裁判",
    "卧底",
)

# 10 分钟时间桶（600 秒），用于判断跨成员称呼是否贯穿整场而非偶然提及
TIME_BUCKET_SEC = 600.0


@dataclass
class SceneClassification:
    bvid: str
    page: int
    up_name: str
    title: str
    old_scene_type: str
    new_scene_type: str
    confidence: float
    detected_members: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    skipped_locked: bool = False


def _match_keywords(text: str, keywords: Sequence[str]) -> List[str]:
    lowered = text.lower()
    return [kw for kw in keywords if kw.lower() in lowered]


def _find_members_in_text(
    text: str,
    members: Sequence[MemberConfig],
    *,
    exclude_up: Optional[str] = None,
) -> List[str]:
    lowered = text.lower()
    found: List[str] = []
    for m in members:
        if exclude_up and m.up_name == exclude_up:
            continue
        # 优先使用长度 >= 2 的正式名与别名，减少单字误判
        for alias in (m.up_name, m.official_name, *m.aliases):
            if len(alias) >= 2 and alias.lower() in lowered:
                found.append(m.up_name)
                break
    return found


def classify_video_scene(
    video: Dict[str, Any],
    subtitles: Sequence[Tuple[float, str]],
    members: Sequence[MemberConfig],
    *,
    fallback_solo: bool = True,
) -> SceneClassification:
    """结合标题元数据与字幕跨成员时间分布，对单场/单分 P 做轻量规则打标."""
    bvid = str(video["bvid"])
    page = int(video["page"])
    up_name = str(video["up_name"])
    title = str(video.get("title") or "")
    part_title = str(video.get("part_title") or "")
    old_scene = str(video.get("scene_type") or "unknown")

    extra_desc = ""
    try:
        meta = json.loads(video.get("extra_meta") or "{}")
        extra_desc = str(meta.get("desc_snippet") or "")
    except Exception:
        pass

    meta_text = f"{title} {part_title} {extra_desc}".strip()
    reasons: List[str] = []

    # 1) 标题/元数据关键词检查
    special_hits = _match_keywords(meta_text, SPECIAL_TITLE_KEYWORDS)
    group_hits = _match_keywords(meta_text, GROUP_TITLE_KEYWORDS)
    duo_hits = _match_keywords(meta_text, DUO_TITLE_KEYWORDS)
    collab_hits = _match_keywords(meta_text, COLLAB_TITLE_KEYWORDS)
    title_other_members = _find_members_in_text(
        f"{title} {part_title}", members, exclude_up=up_name
    )

    if special_hits:
        reasons.append(f"标题命中特别场次词:{','.join(special_hits)}")
    if group_hits:
        reasons.append(f"标题命中团播词:{','.join(group_hits)}")
    if duo_hits:
        reasons.append(f"标题命中双播词:{','.join(duo_hits)}")
    if collab_hits:
        reasons.append(f"标题命中联动词:{','.join(collab_hits)}")
    if title_other_members:
        reasons.append(f"标题出现其他成员:{','.join(title_other_members)}")

    # 2) 字幕跨成员时间分布与互动特征检查
    member_mentions: Dict[str, int] = defaultdict(int)
    member_buckets: Dict[str, Set[int]] = defaultdict(set)
    cue_hits: Dict[str, int] = defaultdict(int)

    # 预编译每个其他成员的别名正则，提升数千行字幕扫描速度
    member_patterns: List[Tuple[str, re.Pattern[str]]] = []
    for m in members:
        if m.up_name == up_name:
            continue
        aliases = sorted(
            {a for a in (m.up_name, m.official_name, *m.aliases) if len(a) >= 2},
            key=len,
            reverse=True,
        )
        if aliases:
            pat = re.compile("|".join(re.escape(a) for a in aliases), re.IGNORECASE)
            member_patterns.append((m.up_name, pat))

    for start_sec, content in subtitles:
        bucket = int(max(0.0, start_sec) // TIME_BUCKET_SEC)
        for m_name, pat in member_patterns:
            if pat.search(content):
                member_mentions[m_name] += 1
                member_buckets[m_name].add(bucket)
        for cue in MULTI_SPEAKER_SUB_CUES:
            if cue in content:
                cue_hits[cue] += 1

    # 活跃同场成员判定：至少在 2 个不同 10 分钟时间桶出现，或累计提及 >= 4 次
    active_sub_members = [
        m_name
        for m_name, cnt in member_mentions.items()
        if len(member_buckets[m_name]) >= 2 or cnt >= 4
    ]
    if active_sub_members:
        detail = ", ".join(
            f"{m}({member_mentions[m]}次/{len(member_buckets[m])}时段)"
            for m in active_sub_members
        )
        reasons.append(f"字幕持续互动成员:{detail}")

    total_cue_hits = sum(cue_hits.values())
    if total_cue_hits > 0:
        top_cues = ",".join(list(cue_hits.keys())[:5])
        reasons.append(f"字幕命中多人互动词:{top_cues}(共{total_cue_hits}次)")

    # 合并识别到的其他成员集合
    detected_others = sorted(set(title_other_members) | set(active_sub_members))

    # 3) 综合决策（优先级：单人【突击】优先判定 > 特别场次 > 团播 > 双播/联动 > 单播/未知）
    new_scene = "unknown"
    confidence = 0.3

    is_tuji_title = bool(_TUJI_TAG_RE.search(title))
    tuji_multi_hits = _match_keywords(f"{title} {part_title}", _TUJI_MULTI_HINTS)

    if is_tuji_title and not title_other_members and not tuji_multi_hits:
        new_scene = "solo"
        confidence = 0.90
        reasons.append("标题含【突击】且未标注双人/多人联机，归为单播")
    elif special_hits:
        new_scene = "special"
        confidence = 0.90
    elif group_hits or len(detected_others) >= 2:
        new_scene = "group"
        if group_hits and len(detected_others) >= 2:
            confidence = 0.95
        elif group_hits:
            confidence = 0.85
        else:
            confidence = 0.78
    elif duo_hits or collab_hits or len(detected_others) == 1:
        if duo_hits or collab_hits or (len(title_other_members) == 1) or (
            len(active_sub_members) == 1
            and (
                len(member_buckets[active_sub_members[0]]) >= 3
                or total_cue_hits >= 2
            )
        ):
            new_scene = "duo"
            confidence = 0.88 if (duo_hits or collab_hits or title_other_members) else 0.75
        else:
            new_scene = "duo"
            confidence = 0.65
    elif sum(cue_hits.get(c, 0) for c in STRONG_GROUP_SUB_CUES) >= 3:
        new_scene = "group"
        confidence = 0.60
    else:
        if fallback_solo and len(subtitles) >= 3:
            new_scene = "solo"
            confidence = 0.70
            reasons.append("未命中多人/团播特征，初筛归为单播")
        else:
            new_scene = "unknown"
            confidence = 0.20
            reasons.append("样本较少且无明显多人特征，保持 unknown")

    return SceneClassification(
        bvid=bvid,
        page=page,
        up_name=up_name,
        title=title,
        old_scene_type=old_scene,
        new_scene_type=new_scene,
        confidence=round(confidence, 2),
        detected_members=detected_others,
        reasons=reasons,
    )


def run_scene_tagging(
    db: CorpusDB,
    config: AppConfig,
    *,
    bvid: Optional[str] = None,
    only_unknown: bool = False,
    fallback_solo: bool = True,
    dry_run: bool = False,
) -> List[SceneClassification]:
    """批量运行轻量场景初筛，默认跳过 scene_locked = 1 的场次."""
    db.init_db()
    with db.session() as conn:
        clauses = ["1=1"]
        params: List[Any] = []
        if bvid:
            clauses.append("bvid = ?")
            params.append(bvid)
        if only_unknown:
            clauses.append("scene_type = 'unknown'")
        sql = f"SELECT * FROM videos WHERE {' AND '.join(clauses)} ORDER BY pub_date DESC, bvid, page"
        videos = [dict(r) for r in conn.execute(sql, params).fetchall()]

    results: List[SceneClassification] = []
    for vid in videos:
        if int(vid.get("scene_locked") or 0) == 1:
            results.append(
                SceneClassification(
                    bvid=vid["bvid"],
                    page=int(vid["page"]),
                    up_name=vid["up_name"],
                    title=vid.get("title", ""),
                    old_scene_type=vid["scene_type"],
                    new_scene_type=vid["scene_type"],
                    confidence=float(vid.get("scene_confidence") or 1.0),
                    reasons=["已人工锁定 scene_locked=1，跳过"],
                    skipped_locked=True,
                )
            )
            continue

        with db.session() as conn:
            sub_rows = conn.execute(
                "SELECT start_sec, content FROM subtitles WHERE bvid = ? AND page = ? ORDER BY start_sec ASC",
                (vid["bvid"], vid["page"]),
            ).fetchall()
        subtitles = [(float(r["start_sec"]), str(r["content"])) for r in sub_rows]

        res = classify_video_scene(
            vid,
            subtitles,
            config.members,
            fallback_solo=fallback_solo,
        )
        results.append(res)

        if not dry_run:
            reason_json = json.dumps(
                {
                    "reasons": res.reasons,
                    "detected_other_members": res.detected_members,
                },
                ensure_ascii=False,
            )
            db.update_scene_type(
                res.bvid,
                res.page,
                res.new_scene_type,
                confidence=res.confidence,
                reason=reason_json,
                lock=False,
            )

    return results
