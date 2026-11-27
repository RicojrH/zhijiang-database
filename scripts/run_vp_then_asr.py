"""串行编排：先 2 进程并行执行声纹发言人标注，完成后自动 2 进程并行执行 GPU ASR 补录."""

from __future__ import annotations

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
    print(f"\n==================== [{time.strftime('%H:%M:%S')}] 启动 {label} ({n_procs} 进程并行) ====================", flush=True)
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
    print(f"==================== [{time.strftime('%H:%M:%S')}] {label} 结束 (exit_codes={rcs}) ====================\n", flush=True)
    return 0 if all(rc == 0 for rc in rcs) else 1


def main() -> int:
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

    # 阶段 1：2 进程并行声纹发言人标注（断点续跑）
    try:
        rc_vp = _run_sharded(
            "VP",
            [
                "run.py",
                "run-voiceprint",
                "--scope",
                "multi_and_watch",
                "--device",
                "auto",
                "--batch-size",
                "64",
                "--limit",
                "0",
                "--threads",
                "4",
                "--prefetch",
                "2",
            ],
            n_procs=2,
            env=env,
        )
    finally:
        _cleanup_vp_tmp()

    if rc_vp != 0:
        print(f"[WARN] 声纹阶段存在非零退出码 (rc={rc_vp})，仍继续启动 2 进程 GPU ASR 补录...", flush=True)

    # 稍等 3 秒确保 GPU 显存完全释放
    time.sleep(3)

    # 阶段 2：2 进程并行 GPU ASR 补录无字幕新分 P
    try:
        rc_asr = _run_sharded(
            "ASR",
            [
                "run.py",
                "backfill-asr",
                "--limit",
                "0",
                "--batch-size",
                "12",
                "--beam-size",
                "1",
                "--chunk-length",
                "10",
                "--workers",
                "1",
                "--prefetch",
                "1",
            ],
            n_procs=2,
            env=env,
        )
    finally:
        _cleanup_asr_tmp()

    # 阶段 3：优化 FTS5 索引与回收 WAL
    print(f"[{time.strftime('%H:%M:%S')}] 正在执行数据库 FTS5 索引优化与 WAL 回收...", flush=True)
    subprocess.run([sys.executable, "-u", "run.py", "optimize-db"], cwd=str(PROJECT_ROOT), env=env, check=False)
    print(f"[{time.strftime('%H:%M:%S')}] 全流程完成 (rc_vp={rc_vp}, rc_asr={rc_asr})", flush=True)
    return 0 if (rc_vp == 0 and rc_asr == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
