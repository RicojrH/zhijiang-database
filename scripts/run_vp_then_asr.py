"""全自动串行编排：
  1. [可选] 3 进程并行声纹标注（multi_and_watch）
  2. 2 进程并行 GPU ASR 补录无字幕分 P
  3. 专有名词纠错 + 全库场景标签复检（tag-scenes --only-unknown + 二创计画合集核对）
  4. 3 进程并行声纹标注（先跑 ASR 后新产生的多人场次 / 电影电台回：--scope multi_and_watch）
  5. 3 进程并行全量不重复声纹标注（剩余全部未标注场次：--scope all，绝不重复跑已 ok 场次）
  6. 压缩优化 FTS5 索引与 WAL
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _cleanup_vp_tmp() -> None:
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


def _cleanup_asr_tmp() -> None:
    tmp_base = PROJECT_ROOT / "data" / "tmp_asr"
    if tmp_base.exists():
        for d in tmp_base.glob("zj_asr_*"):
            shutil.rmtree(d, ignore_errors=True)


def _run_sharded(label: str, base_args: List[str], n_procs: int, env: dict[str, str]) -> int:
    py = sys.executable
    procs: List[subprocess.Popen] = []
    print(
        f"\n==================== [{time.strftime('%H:%M:%S')}] 启动 {label} ({n_procs} 进程并行) ====================",
        flush=True,
    )
    for idx in range(n_procs):
        cmd = [py, "-u", *base_args, "--shard-index", str(idx), "--shard-total", str(n_procs)]
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
        procs.append(proc)

    def _pump(p: subprocess.Popen, shard: int) -> None:
        if not p.stdout:
            return
        prefix = f"[{label}#{shard + 1}] "
        for line in p.stdout:
            txt = line.rstrip()
            if txt:
                print(f"{prefix}{txt}", flush=True)

    threads = [threading.Thread(target=_pump, args=(p, i), daemon=True) for i, p in enumerate(procs)]
    for t in threads:
        t.start()

    rcs = [p.wait() for p in procs]
    for t in threads:
        t.join(timeout=5)
    print(
        f"==================== [{time.strftime('%H:%M:%S')}] {label} 结束 (exit_codes={rcs}) ====================\n",
        flush=True,
    )
    return 0 if all(rc == 0 for rc in rcs) else 1


def _run_step(label: str, cmd: List[str], env: dict[str, str]) -> int:
    print(f"\n==================== [{time.strftime('%H:%M:%S')}] {label} ====================", flush=True)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)
    print(f"==================== [{time.strftime('%H:%M:%S')}] {label} 结束 (rc={proc.returncode}) ====================\n", flush=True)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="声纹 -> ASR -> 标签复检 -> 增量多人声纹 -> 全量声纹编排脚本")
    parser.add_argument("--skip-initial-vp", action="store_true", help="跳过阶段 1 初始声纹，直接从阶段 2 ASR 开始")
    parser.add_argument("--vp-procs", type=int, default=3, help="声纹并行进程数（默认 3）")
    parser.add_argument("--asr-procs", type=int, default=3, help="ASR 并行进程数（默认 3）")
    parser.add_argument("--asr-prefetch", type=int, default=3, help="每 ASR 进程并发预取下载线程数（默认 3）")
    parser.add_argument("--vp-prefetch", type=int, default=3, help="每声纹进程并发预取下载线程数（默认 3）")
    args = parser.parse_args()

    env = os.environ.copy()
    hf_dir = PROJECT_ROOT / "data" / "cache" / "huggingface"
    tmp_dir = PROJECT_ROOT / "data" / "tmp_asr"
    hf_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(hf_dir)
    env["TMP"] = str(tmp_dir)
    env["TEMP"] = str(tmp_dir)
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_CACHE_MAXSIZE"] = "2147483648"
    py = sys.executable

    # 阶段 1（可选）：初始 multi_and_watch 声纹标注
    if not args.skip_initial_vp:
        try:
            _run_sharded(
                "VP1",
                [
                    "run.py", "run-voiceprint",
                    "--scope", "multi_and_watch",
                    "--device", "auto",
                    "--batch-size", "64",
                    "--limit", "0",
                    "--threads", "4",
                    "--prefetch", "2",
                ],
                n_procs=args.vp_procs,
                env=env,
            )
        finally:
            _cleanup_vp_tmp()
        time.sleep(3)

    # 阶段 2：多进程并行 GPU ASR 补录全部无字幕新分 P（含 1 次断流自动重试）
    asr_base_cmd = [
        "run.py", "backfill-asr",
        "--limit", "0",
        "--batch-size", "12",
        "--beam-size", "1",
        "--chunk-length", "10",
        "--workers", "1",
        "--prefetch", str(args.asr_prefetch),
    ]
    try:
        rc_asr = _run_sharded("ASR", asr_base_cmd, n_procs=args.asr_procs, env=env)
    finally:
        _cleanup_asr_tmp()
    time.sleep(2)
    # 若存在偶发 B 站 CDN 断流导致的残留 no_subtitle 分 P，自动补跑 1 轮重试
    try:
        rc_asr_retry = _run_sharded("ASR-RETRY", asr_base_cmd, n_procs=min(2, args.asr_procs), env=env)
        if rc_asr != 0 and rc_asr_retry == 0:
            rc_asr = 0
    finally:
        _cleanup_asr_tmp()
    time.sleep(3)

    # 阶段 3：专有名词纠错 + 全库场景标签复检（含新 ASR 场次与二创计画合集核对）
    _run_step("专有名词纠错标准化", [py, "-u", "run.py", "normalize-corpus"], env)
    _run_step("场景初筛补全 unknown", [py, "-u", "run.py", "tag-scenes", "--only-unknown"], env)
    _run_step("二创计画合集标签复核", [py, "-u", "scripts/sync_archives.py", "--only-tags"], env)

    # 阶段 4：先对 ASR 后新产生的多人场次 / 电影电台回进行 3 进程声纹识别（绝不重复跑已 ok 场次）
    try:
        rc_vp_new_multi = _run_sharded(
            "VP-NEW-MULTI",
            [
                "run.py", "run-voiceprint",
                "--scope", "multi_and_watch",
                "--device", "auto",
                "--batch-size", "64",
                "--limit", "0",
                "--threads", "4",
                "--prefetch", str(args.vp_prefetch),
            ],
            n_procs=args.vp_procs,
            env=env,
        )
    finally:
        _cleanup_vp_tmp()
    time.sleep(3)

    # 阶段 5：再对全库剩余全部未标注场次（单播等）进行 3 进程全量不重复声纹识别
    try:
        rc_vp_all = _run_sharded(
            "VP-ALL-REM",
            [
                "run.py", "run-voiceprint",
                "--scope", "all",
                "--device", "auto",
                "--batch-size", "64",
                "--limit", "0",
                "--threads", "4",
                "--prefetch", str(args.vp_prefetch),
            ],
            n_procs=args.vp_procs,
            env=env,
        )
    finally:
        _cleanup_vp_tmp()

    # 阶段 6：优化 FTS5 索引与回收 WAL
    _run_step("压缩优化 FTS5 索引与 WAL", [py, "-u", "run.py", "optimize-db"], env)
    print(
        f"[{time.strftime('%H:%M:%S')}] 全流程完成 "
        f"(rc_asr={rc_asr}, rc_vp_new_multi={rc_vp_new_multi}, rc_vp_all={rc_vp_all})",
        flush=True,
    )
    return 0 if (rc_asr == 0 and rc_vp_new_multi == 0 and rc_vp_all == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
