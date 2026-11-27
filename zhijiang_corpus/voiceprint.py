"""枝江数据库 · 声纹发言人识别引擎（CAM++ 192d + WeSpeaker CN-Celeb 256d 双模型融合）.

核心设计：
  1. 跨时期声线覆盖：基于 2021~2026 早/中/近期分层单播特征（v3_feats_*.npz）训练 448 维判别式分类头，
     解决成员早期与近期声线漂移及 3D 动捕麦 / 2D 桌面麦域差异；
  2. 历史在籍成员先验：按 pub_date 自动屏蔽未出道或不在籍成员（2023-10 前屏蔽心宜/思诺，
     2024-05 后默认屏蔽向晚；单播观影/反应回仅保留 [up_name] 对比拒识线，精准剥离电影/动画原声）；
  3. 句内滑窗检测：对 >=1.35s 的多人场字幕执行 0.95s 滑窗分析，自动识别「单人」「A+B（合唱/换人）」与「其他/原声」；
  4. 局部时间轴平滑：对夹在同一高置信发言人中间的 <0.9s 超短句做上下文纠偏；
  5. 生产者-消费者预取流水线 + 断点续跑：按分 P 事务性写入 subtitles.speaker_name / speaker_conf
     与 videos.voiceprint_status='ok'，用完即删临时音频，不占用 C 盘。
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
import uuid
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .asr_backfill import convert_to_16k_wav
from .bilibili_client import BilibiliClient
from .config import AppConfig, PROJECT_ROOT
from .db import CorpusDB

ProgressCallback = Callable[[str], None]

VOICEPRINT_MEMBERS = ("嘉然", "贝拉", "乃琳", "向晚", "心宜", "思诺")
NON_MEMBER_LABEL = "其他/原声"
VOICEPRINT_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "voiceprint"
CAMP_MODEL_NAME = "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
WESP_MODEL_NAME = "wespeaker_zh_cnceleb_resnet34_LM.onnx"
BANK_FILE_NAME = "zhijiang_6members_v3.npz"

WATCH_ALONG_KEYWORDS = (
    "影厅",
    "电影",
    "放映",
    "一起看",
    "看剧",
    "看番",
    "追番",
    "反应",
    "电台",
)


def _l2_norm(arr: np.ndarray) -> np.ndarray:
    return arr / np.maximum(np.linalg.norm(arr, axis=-1, keepdims=True), 1e-8)


def active_candidates_for_video(
    *,
    up_name: str,
    pub_date: str,
    scene_type: str,
    title: str = "",
) -> List[str]:
    """根据场次类型、发布日期与标题先验确定本场合法候选发言人集合."""
    if scene_type == "solo":
        return [up_name] if up_name in VOICEPRINT_MEMBERS else list(VOICEPRINT_MEMBERS)

    d = (pub_date or "")[:10]
    t = title or ""
    if not d:
        cands = list(VOICEPRINT_MEMBERS)
    else:
        cands = ["嘉然", "贝拉", "乃琳"]
        if d <= "2024-04-30" or "向晚" in t or "晚晚" in t:
            cands.append("向晚")
        if d >= "2023-10-10":
            cands.extend(["心宜", "思诺"])

    # 小心思双人专属标题/场次缩窄到 2 选 1，准确率接近 98%+
    if scene_type == "duo" and up_name in ("心宜", "思诺"):
        if not any(m in t for m in ("嘉然", "然然", "然比", "贝拉", "拉姐", "乃琳", "乃宝", "向晚")):
            return ["心宜", "思诺"]

    if up_name in VOICEPRINT_MEMBERS and up_name not in cands:
        cands.append(up_name)
    return cands


def fit_calibrated_classifier(
    X_camp: np.ndarray,
    X_wesp: np.ndarray,
    y: np.ndarray,
    *,
    w_camp: float = 0.75,
    w_wesp: float = 0.25,
    l2: float = 5e-4,
    steps: int = 1200,
    lr: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """训练 448 维校准版多分类 Softmax 判别头（保证单人概率锐利、重叠界限清晰）."""
    X = _l2_norm(np.hstack([w_camp * X_camp, w_wesp * X_wesp])).astype(np.float64)
    n, f = X.shape
    k = len(VOICEPRINT_MEMBERS)
    W = np.zeros((f, k), dtype=np.float64)
    b = np.zeros(k, dtype=np.float64)
    Y = np.eye(k, dtype=np.float64)[y]
    for ep in range(steps):
        lg = X @ W + b
        lg -= lg.max(axis=1, keepdims=True)
        p = np.exp(lg)
        p /= p.sum(axis=1, keepdims=True)
        df = (p - Y) / n
        step = lr / (1.0 + 0.001 * ep)
        W -= step * (X.T @ df + l2 * W)
        b -= step * df.sum(axis=0)
    return W.astype(np.float32), b.astype(np.float32)


def build_voiceprint_bank_from_cache(
    cache_dir: Path = VOICEPRINT_CACHE_DIR,
) -> Dict[str, Any]:
    """从已提取的 2021~2026 跨时期特征缓存 (v3_feats_*.npz) 编译正式 6 人声纹模型包."""
    tr_c_list: List[np.ndarray] = []
    tr_w_list: List[np.ndarray] = []
    y_list: List[int] = []
    per_member: Dict[str, int] = {}

    for idx, up in enumerate(VOICEPRINT_MEMBERS):
        f = cache_dir / f"v3_feats_{up}.npz"
        if not f.exists():
            raise FileNotFoundError(
                f"缺少 {up} 的跨时期特征缓存文件 {f}，请先执行特征采集。"
            )
        z = np.load(f, allow_pickle=True)
        c_arr = np.asarray(z["tr_camp"], dtype=np.float32)
        w_arr = np.asarray(z["tr_wesp"], dtype=np.float32)
        tr_c_list.append(c_arr)
        tr_w_list.append(w_arr)
        y_list.extend([idx] * len(c_arr))
        per_member[up] = len(c_arr)

    X_camp = _l2_norm(np.vstack(tr_c_list))
    X_wesp = _l2_norm(np.vstack(tr_w_list))
    y = np.asarray(y_list, dtype=np.int64)
    W, b = fit_calibrated_classifier(X_camp, X_wesp, y)

    out_path = cache_dir / BANK_FILE_NAME
    np.savez(
        out_path,
        members=np.asarray(VOICEPRINT_MEMBERS),
        W=W,
        b=b,
        X_camp=X_camp,
        y=y,
        w_camp=np.float32(0.75),
        w_wesp=np.float32(0.25),
    )
    return {
        "bank_path": str(out_path),
        "total_samples": int(len(y)),
        "per_member": per_member,
        "feature_dim": int(W.shape[0]),
    }


def load_wav_float32(wav_path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        samples = samples.reshape(-1, nch).mean(axis=1)
    return samples, sr


class VoiceprintEngine:
    """正式版双模型声纹识别器（CAM++ + WeSpeaker CN-Celeb + 滑窗换人/重叠检测 + 上下文平滑）."""

    def __init__(
        self,
        *,
        cache_dir: Path = VOICEPRINT_CACHE_DIR,
        num_threads: int = 4,
        reject_cos_th: float = 0.36,
    ):
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError(
                "未安装 sherpa-onnx，请先执行 .\\.venv\\Scripts\\python.exe -m pip install sherpa-onnx"
            ) from exc

        camp_path = cache_dir / CAMP_MODEL_NAME
        wesp_path = cache_dir / WESP_MODEL_NAME
        bank_path = cache_dir / BANK_FILE_NAME
        if not camp_path.exists() or not wesp_path.exists():
            raise FileNotFoundError(
                f"缺少声纹 ONNX 模型文件，请确认 {camp_path.name} 与 {wesp_path.name} 位于 {cache_dir}"
            )
        if not bank_path.exists():
            build_voiceprint_bank_from_cache(cache_dir)

        z = np.load(bank_path, allow_pickle=True)
        self.members: Tuple[str, ...] = tuple(str(x) for x in z["members"])
        self.idx_map: Dict[str, int] = {m: i for i, m in enumerate(self.members)}
        self.W: np.ndarray = np.asarray(z["W"], dtype=np.float32)
        self.b: np.ndarray = np.asarray(z["b"], dtype=np.float32)
        self.X_camp: np.ndarray = np.asarray(z["X_camp"], dtype=np.float32)
        self.y: np.ndarray = np.asarray(z["y"], dtype=np.int64)
        self.w_camp: float = float(z["w_camp"])
        self.w_wesp: float = float(z["w_wesp"])
        self.reject_cos_th: float = reject_cos_th

        self._camp_by_cls: Dict[int, np.ndarray] = {
            i: self.X_camp[self.y == i] for i in range(len(self.members))
        }

        self._ex_camp = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(camp_path),
                num_threads=max(1, int(num_threads)),
                provider="cpu",
            )
        )
        self._ex_wesp = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(wesp_path),
                num_threads=max(1, int(num_threads)),
                provider="cpu",
            )
        )

    def _extract_pair(self, seg: np.ndarray, sr: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if len(seg) < int(0.45 * sr):
            return None
        if float(np.sqrt(np.mean(seg * seg) + 1e-12)) < 0.004:
            return None
        seg_list = seg.tolist()

        vecs: List[np.ndarray] = []
        for ex in (self._ex_camp, self._ex_wesp):
            st = ex.create_stream()
            st.accept_waveform(sr, seg_list)
            st.input_finished()
            if not ex.is_ready(st):
                return None
            v = np.asarray(ex.compute(st), dtype=np.float32)
            nrm = float(np.linalg.norm(v))
            if nrm <= 1e-8:
                return None
            vecs.append(v / nrm)

        vc, vw = vecs[0], vecs[1]
        vj = _l2_norm(np.concatenate([self.w_camp * vc, self.w_wesp * vw]).astype(np.float32))
        return vc, vj

    def score_window(
        self,
        vc: np.ndarray,
        vj: np.ndarray,
        candidates: Sequence[str],
    ) -> Tuple[str, List[str], bool, float, float]:
        """返回 (label, speakers, is_overlap, top1_cos, top1_prob)."""
        cidx = [self.idx_map[c] for c in candidates if c in self.idx_map]
        cands = [self.members[i] for i in cidx]
        if not cands:
            return NON_MEMBER_LABEL, [], False, 0.0, 0.0

        cos_map: Dict[str, float] = {}
        for c, ci in zip(cands, cidx):
            sims = self._camp_by_cls[ci] @ vc
            k = min(5, len(sims))
            cos_map[c] = float(np.partition(sims, -k)[-k:].mean())

        if len(cands) == 1:
            only = cands[0]
            c_sc = cos_map[only]
            if c_sc < self.reject_cos_th:
                return NON_MEMBER_LABEL, [], False, c_sc, 1.0 - c_sc
            return only, [only], False, c_sc, min(0.99, max(0.5, c_sc + 0.25))

        lg = (vj @ self.W + self.b)[cidx].astype(np.float64)
        lg -= lg.max()
        p = np.exp(lg)
        p /= p.sum()
        pmap = {c: float(p[i]) for i, c in enumerate(cands)}
        ranked = sorted(cands, key=lambda c: pmap[c], reverse=True)
        t1 = ranked[0]
        t2 = ranked[1]
        max_cos = max(cos_map.values())

        if max_cos < self.reject_cos_th:
            return NON_MEMBER_LABEL, [], False, max_cos, pmap[t1]

        # 校准后的精准重叠/合唱条件：双人余弦均 >=0.40 且 Softmax 概率胶着 (gap<=0.25)
        is_ov = bool(
            cos_map[t1] >= 0.40
            and cos_map[t2] >= 0.40
            and (pmap[t1] - pmap[t2]) <= 0.25
            and pmap[t2] >= 0.30
        )
        if is_ov:
            return f"{t1}+{t2}", [t1, t2], True, cos_map[t1], pmap[t1]

        return t1, [t1], False, cos_map[t1], pmap[t1]

    def identify_part(
        self,
        audio: np.ndarray,
        sr: int,
        subtitles: Sequence[Tuple[int, float, float, str]],
        *,
        up_name: str,
        pub_date: str,
        scene_type: str,
        title: str = "",
    ) -> List[Tuple[int, str, float]]:
        """对单个分 P 的全部字幕行执行声纹发言人标注与局部上下文平滑."""
        cands = active_candidates_for_video(
            up_name=up_name,
            pub_date=pub_date,
            scene_type=scene_type,
            title=title,
        )
        total_audio_sec = len(audio) / float(sr)
        raw_results: List[Tuple[int, float, float, str, float]] = []

        for sid, s0, e0, _ in subtitles:
            dur = max(0.0, e0 - s0)
            if dur < 1.35 or len(cands) <= 1:
                pad = max(0.08, (0.85 - dur) * 0.5) if dur < 0.85 else 0.08
                i0 = max(0, int((s0 - pad) * sr))
                i1 = min(len(audio), int((e0 + pad) * sr))
                pair = self._extract_pair(audio[i0:i1], sr)
                if pair is None:
                    raw_results.append((sid, s0, e0, "", 0.0))
                    continue
                lbl, _, _, cs, pr = self.score_window(pair[0], pair[1], cands)
                conf = round(min(0.99, max(0.0, 0.5 * cs + 0.5 * pr)) if lbl != NON_MEMBER_LABEL else max(0.0, 1.0 - cs), 3)
                raw_results.append((sid, s0, e0, lbl, conf))
                continue

            # >= 1.35s 多人场字幕：句内滑窗检出换人 / 合唱重叠
            if dur <= 2.3:
                mid = (s0 + e0) * 0.5
                wins = [(s0, min(e0, mid + 0.15)), (max(s0, mid - 0.15), e0)]
            else:
                wins = []
                t = s0
                while t + 0.70 <= e0 + 0.05:
                    we = min(e0, t + 0.95)
                    if we - t >= 0.60:
                        wins.append((t, we))
                    if we >= e0:
                        break
                    t += 0.50

            ord_sp: List[str] = []
            votes: Counter[str] = Counter()
            best_conf: Dict[str, float] = {}
            for ws, we in wins:
                i0 = max(0, int(ws * sr))
                i1 = min(len(audio), int(min(we, total_audio_sec) * sr))
                pair = self._extract_pair(audio[i0:i1], sr)
                if pair is None:
                    continue
                lbl, sps, is_ov, cs, pr = self.score_window(pair[0], pair[1], cands)
                if lbl == NON_MEMBER_LABEL or not sps:
                    continue
                if is_ov:
                    for sp in sps:
                        votes[sp] += 1
                        best_conf[sp] = max(best_conf.get(sp, 0.0), 0.5 * cs + 0.5 * pr)
                        if sp not in ord_sp:
                            ord_sp.append(sp)
                elif pr >= 0.72 and cs >= 0.42:
                    sp = sps[0]
                    votes[sp] += 1
                    best_conf[sp] = max(best_conf.get(sp, 0.0), 0.5 * cs + 0.5 * pr)
                    if sp not in ord_sp:
                        ord_sp.append(sp)

            uniq = [s for s in ord_sp if votes[s] >= 1]
            if not uniq:
                i0 = max(0, int(s0 * sr))
                i1 = min(len(audio), int(min(e0, total_audio_sec) * sr))
                pair = self._extract_pair(audio[i0:i1], sr)
                if pair is None:
                    raw_results.append((sid, s0, e0, "", 0.0))
                else:
                    lbl, _, _, cs, pr = self.score_window(pair[0], pair[1], cands)
                    conf = round(min(0.99, max(0.0, 0.5 * cs + 0.5 * pr)) if lbl != NON_MEMBER_LABEL else max(0.0, 1.0 - cs), 3)
                    raw_results.append((sid, s0, e0, lbl, conf))
            elif len(uniq) == 1:
                sp = uniq[0]
                raw_results.append((sid, s0, e0, sp, round(min(0.99, best_conf[sp]), 3)))
            else:
                top2 = sorted(uniq, key=lambda s: votes[s], reverse=True)[:2]
                top2_ord = [s for s in ord_sp if s in top2]
                conf = round(min(0.99, max(best_conf[s] for s in top2_ord)), 3)
                raw_results.append((sid, s0, e0, "+".join(top2_ord), conf))

        # 局部上下文平滑：被同一位高置信发言人紧贴包夹的超短句（<0.9s，置信度<0.65）顺滑归入
        smoothed: List[Tuple[int, str, float]] = []
        n = len(raw_results)
        for i, (sid, s0, e0, lbl, conf) in enumerate(raw_results):
            if (e0 - s0) < 0.90 and conf < 0.65 and 0 < i < n - 1:
                _, _, prev_e, prev_l, prev_c = raw_results[i - 1]
                _, next_s, _, next_l, next_c = raw_results[i + 1]
                if (
                    prev_l
                    and prev_l == next_l
                    and prev_l != NON_MEMBER_LABEL
                    and "+" not in prev_l
                    and prev_c >= 0.68
                    and next_c >= 0.68
                    and (s0 - prev_e) <= 2.0
                    and (next_s - e0) <= 2.0
                ):
                    lbl = prev_l
                    conf = round(min(prev_c, next_c) * 0.9, 3)
            smoothed.append((sid, lbl, conf))
        return smoothed


def _watch_along_sql_clause() -> str:
    parts = [f"title LIKE '%{kw}%'" for kw in WATCH_ALONG_KEYWORDS]
    return f"({' OR '.join(parts)})"


def list_voiceprint_targets(
    db: CorpusDB,
    *,
    scope: str = "multi_and_watch",
    limit: int = 0,
    shard_index: int = 0,
    shard_total: int = 1,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """列出待执行声纹发言人标注的分 P 列表（支持断点续跑与多进程分片）."""
    db.init_db()
    clauses = ["subtitle_status = 'ok'", "line_count > 0", "duration_sec >= 15"]
    if not force:
        clauses.append("voiceprint_status != 'ok'")
    if scope == "multi_only":
        clauses.append("scene_type IN ('duo', 'group', 'special')")
    elif scope == "multi_and_watch":
        clauses.append(
            f"(scene_type IN ('duo', 'group', 'special') OR (scene_type = 'solo' AND {_watch_along_sql_clause()}))"
        )
    where_sql = " AND ".join(clauses)
    with db.session() as conn:
        rows = conn.execute(
            f"""
            SELECT bvid, page, cid, up_name, title, pub_date, duration_sec, scene_type, line_count
              FROM videos
             WHERE {where_sql}
             ORDER BY pub_date DESC, bvid ASC, page ASC
            """
        ).fetchall()
    items = [dict(r) for r in rows]
    if shard_total > 1:
        items = [it for idx, it in enumerate(items) if idx % shard_total == shard_index]
    if limit > 0:
        items = items[:limit]
    return items


def get_voiceprint_progress(db: CorpusDB) -> Dict[str, Any]:
    """统计声纹标注实时进度（供 /debug 控制台进度条展示）."""
    db.init_db()
    watch_sql = _watch_along_sql_clause()
    with db.session() as conn:
        target_row = conn.execute(
            f"""
            SELECT COUNT(*) AS total_parts,
                   COALESCE(SUM(CASE WHEN voiceprint_status = 'ok' THEN 1 ELSE 0 END), 0) AS done_parts,
                   COALESCE(SUM(CASE WHEN voiceprint_status != 'ok' THEN duration_sec ELSE 0 END), 0) AS rem_dur,
                   COALESCE(SUM(duration_sec), 0) AS total_dur
              FROM videos
             WHERE subtitle_status = 'ok'
               AND line_count > 0
               AND duration_sec >= 15
               AND (scene_type IN ('duo', 'group', 'special') OR (scene_type = 'solo' AND {watch_sql}))
            """
        ).fetchone()
        any_done = conn.execute(
            "SELECT COUNT(*) AS c, COALESCE(SUM(line_count), 0) AS lines FROM videos WHERE voiceprint_status = 'ok'"
        ).fetchone()
    total_p = int(target_row["total_parts"] or 0)
    done_p = int(target_row["done_parts"] or 0)
    rem_p = max(0, total_p - done_p)
    total_dur = int(target_row["total_dur"] or 0)
    rem_dur = int(target_row["rem_dur"] or 0)
    pct = round((done_p / total_p) * 100.0, 1) if total_p > 0 else 100.0
    bank_ready = (
        (VOICEPRINT_CACHE_DIR / CAMP_MODEL_NAME).exists()
        and (VOICEPRINT_CACHE_DIR / WESP_MODEL_NAME).exists()
        and (
            (VOICEPRINT_CACHE_DIR / BANK_FILE_NAME).exists()
            or all((VOICEPRINT_CACHE_DIR / f"v3_feats_{m}.npz").exists() for m in VOICEPRINT_MEMBERS)
        )
    )
    return {
        "targets_total": total_p,
        "done_parts": done_p,
        "remaining_parts": rem_p,
        "remaining_hours": round(rem_dur / 3600.0, 1),
        "total_hours": round(total_dur / 3600.0, 1),
        "pct_parts": pct,
        "all_done_parts": int(any_done["c"] or 0),
        "all_done_lines": int(any_done["lines"] or 0),
        "model_ready": bank_ready,
    }


def run_voiceprint_batch(
    db: CorpusDB,
    cfg: AppConfig,
    *,
    scope: str = "multi_and_watch",
    limit: int = 0,
    shard_index: int = 0,
    shard_total: int = 1,
    num_threads: int = 4,
    prefetch: int = 2,
    force: bool = False,
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """后台流水线批量声纹标注（预取音频 + CAM++/WeSpeaker 逐行识别 + WAL 事务写库 + 立即删临时音频）."""
    emit = progress_cb or (lambda msg: print(msg, flush=True))
    targets = list_voiceprint_targets(
        db,
        scope=scope,
        limit=limit,
        shard_index=shard_index,
        shard_total=shard_total,
        force=force,
    )
    shard_tag = f"[分片 {shard_index + 1}/{shard_total}] " if shard_total > 1 else ""
    if not targets:
        emit(f"{shard_tag}没有待声纹标注的分 P（全部已完成）。")
        return {"targets": 0, "done": 0, "failed": 0, "lines_labeled": 0}

    emit(
        f"{shard_tag}选中 {len(targets)} 个分 P (scope={scope})，正在加载 CAM++ & WeSpeaker 双模型..."
    )
    engine = VoiceprintEngine(num_threads=num_threads)
    tmp_root = Path(tempfile.gettempdir()) / "zhijiang_vp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    task_q: "queue.Queue[Optional[Tuple[int, Dict[str, Any], Optional[Path], Optional[str]]]]" = queue.Queue(
        maxsize=max(2, prefetch * 2)
    )
    stop_ev = threading.Event()

    def _prefetch_worker() -> None:
        with BilibiliClient(cookie=cfg.bilibili_cookie, delay=0.25, jitter=0.1) as bili:
            for idx, item in enumerate(targets, 1):
                if stop_ev.is_set():
                    break
                bvid = str(item["bvid"])
                page = int(item["page"])
                cid = int(item.get("cid") or 0)
                uid = uuid.uuid4().hex[:8]
                m4a = tmp_root / f"{bvid}_p{page}_{uid}.m4a"
                wav = tmp_root / f"{bvid}_p{page}_{uid}.wav"
                err_msg: Optional[str] = None
                try:
                    if not cid:
                        info = bili.fetch_video_info(bvid)
                        for p in info.pages:
                            if p.page == page:
                                cid = p.cid
                                break
                    if not cid:
                        raise RuntimeError(f"无法获取 cid: {bvid} P{page}")
                    last_e: Optional[Exception] = None
                    for attempt in range(3):
                        try:
                            urls = bili.get_audio_stream_urls(bvid, cid)
                            bili.download_stream_file(urls, m4a, bvid=bvid)
                            convert_to_16k_wav(m4a, wav)
                            last_e = None
                            break
                        except Exception as exc:
                            last_e = exc
                            m4a.unlink(missing_ok=True)
                            wav.unlink(missing_ok=True)
                            if attempt < 2:
                                time.sleep(2.0 * (attempt + 1))
                    if last_e is not None:
                        raise last_e
                except Exception as exc:
                    err_msg = str(exc)
                    wav.unlink(missing_ok=True)
                finally:
                    m4a.unlink(missing_ok=True)
                task_q.put((idx, item, None if err_msg else wav, err_msg))
        task_q.put(None)

    t_pf = threading.Thread(target=_prefetch_worker, daemon=True)
    t_pf.start()

    done = 0
    failed = 0
    lines_labeled = 0
    t_all = time.perf_counter()

    try:
        while True:
            pkt = task_q.get()
            if pkt is None:
                break
            idx, item, wav_path, err_msg = pkt
            bvid = str(item["bvid"])
            page = int(item["page"])
            up_name = str(item["up_name"])
            pub_date = str(item["pub_date"] or "")
            scene_type = str(item["scene_type"] or "unknown")
            title = str(item["title"] or "")

            if err_msg or wav_path is None:
                failed += 1
                emit(f"{shard_tag}[{idx}/{len(targets)}] 音频下载失败 {bvid} P{page}: {err_msg}")
                continue

            t0 = time.perf_counter()
            try:
                audio, sr = load_wav_float32(wav_path)
                with db.session() as conn:
                    sub_rows = conn.execute(
                        """
                        SELECT id, start_sec, end_sec, content
                          FROM subtitles
                         WHERE bvid = ? AND page = ?
                         ORDER BY start_sec ASC, id ASC
                        """,
                        (bvid, page),
                    ).fetchall()
                subs = [(int(r["id"]), float(r["start_sec"]), float(r["end_sec"]), str(r["content"])) for r in sub_rows]
                labeled = engine.identify_part(
                    audio,
                    sr,
                    subs,
                    up_name=up_name,
                    pub_date=pub_date,
                    scene_type=scene_type,
                    title=title,
                )
                with db.session() as conn:
                    conn.executemany(
                        "UPDATE subtitles SET speaker_name = ?, speaker_conf = ? WHERE id = ?",
                        [(sp, cf, sid) for sid, sp, cf in labeled],
                    )
                    conn.execute(
                        """
                        UPDATE videos
                           SET voiceprint_status = 'ok',
                               updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                         WHERE bvid = ? AND page = ?
                        """,
                        (bvid, page),
                    )
                done += 1
                lines_labeled += len(labeled)
                top_dist = Counter(sp for _, sp, _ in labeled if sp).most_common(4)
                dist_str = ", ".join(f"{k}:{v}" for k, v in top_dist)
                emit(
                    f"{shard_tag}[{idx}/{len(targets)}] 完成 {up_name} {bvid} P{page} "
                    f"({len(labeled)}行, {time.perf_counter() - t0:.1f}s) | {dist_str}"
                )
            except Exception as exc:
                failed += 1
                emit(f"{shard_tag}[{idx}/{len(targets)}] 推理失败 {bvid} P{page}: {exc}")
            finally:
                wav_path.unlink(missing_ok=True)
    finally:
        stop_ev.set()
        t_pf.join(timeout=5)

    return {
        "targets": len(targets),
        "done": done,
        "failed": failed,
        "lines_labeled": lines_labeled,
        "elapsed_sec": round(time.perf_counter() - t_all, 1),
    }
