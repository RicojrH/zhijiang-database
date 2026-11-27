"""本地运维后台的云端同步、公告发布、纯标签热更新与 GPU ASR 进度管理模块.

安全设计：
  - 仅在本地管理模式（backend/routers/admin.py）中被调用，云端只读模式永不挂载。
  - SSH 密码仅在单次请求内存中传递，绝不写入任何磁盘文件、日志或 Git 仓库。
  - 服务器地址配置保存在本地 .cloud_sync.env（已在 .gitignore 中排除）。
  - 支持一键安装 ~/.ssh/id_ed25519 公钥到云端，实现长期免密同步。
"""

from __future__ import annotations

import json
import os
import posixpath
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from .config import PROJECT_ROOT, load_config
from .db import CorpusDB
from .ingest import ingest_from_up
from .normalizer import run_corpus_normalization
from .scene_tagger import run_scene_tagging
from .sync_delta import export_delta, export_scene_tags

LogCallback = Callable[[str], None]

CLOUD_CONF_PATH = PROJECT_ROOT / ".cloud_sync.env"
ANNOUNCEMENTS_PATH = PROJECT_ROOT / "config" / "announcements.json"
TMP_SYNC_DIR = PROJECT_ROOT / "data" / "tmp_asr"


def _default_ssh_key_paths() -> List[Path]:
    ssh_dir = Path.home() / ".ssh"
    return [ssh_dir / "id_ed25519", ssh_dir / "id_rsa"]


def find_local_ssh_key() -> Optional[Path]:
    for p in _default_ssh_key_paths():
        if p.exists():
            return p
    return None


def load_cloud_config() -> Dict[str, Any]:
    conf: Dict[str, Any] = {
        "cloud_host": "",
        "cloud_app_dir": "/opt/zhijiang",
        "public_url": "https://zjdb.mprlzy.top",
        "ingest_limit": 10,
    }
    if CLOUD_CONF_PATH.exists():
        for raw_line in CLOUD_CONF_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().upper()
            v = v.strip()
            if k == "CLOUD_HOST":
                conf["cloud_host"] = v
            elif k == "CLOUD_APP_DIR" and v:
                conf["cloud_app_dir"] = v
            elif k == "PUBLIC_URL" and v:
                conf["public_url"] = v.rstrip("/")
            elif k == "INGEST_LIMIT" and v.isdigit():
                conf["ingest_limit"] = int(v)
    key_path = find_local_ssh_key()
    conf["has_ssh_key"] = key_path is not None
    conf["ssh_key_path"] = str(key_path) if key_path else ""
    return conf


def save_cloud_config(
    *,
    cloud_host: str,
    cloud_app_dir: str = "/opt/zhijiang",
    public_url: str = "https://zjdb.mprlzy.top",
    ingest_limit: int = 10,
) -> Dict[str, Any]:
    host = (cloud_host or "").strip()
    if host and "@" not in host:
        host = f"root@{host}"
    app_dir = (cloud_app_dir or "/opt/zhijiang").strip()
    pub = (public_url or "https://zjdb.mprlzy.top").strip().rstrip("/")
    lim = max(1, min(500, int(ingest_limit or 10)))
    lines = [
        f"CLOUD_HOST={host}",
        f"CLOUD_APP_DIR={app_dir}",
        f"PUBLIC_URL={pub}",
        f"INGEST_LIMIT={lim}",
    ]
    CLOUD_CONF_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_cloud_config()


def load_announcements_file() -> List[Dict[str, Any]]:
    if not ANNOUNCEMENTS_PATH.exists():
        return []
    try:
        raw = json.loads(ANNOUNCEMENTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = raw.get("items", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    out: List[Dict[str, Any]] = []
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        level = str(it.get("level", "info")).strip().lower()
        if level not in ("info", "change", "warn"):
            level = "info"
        out.append(
            {
                "id": str(it.get("id") or f"ann-{idx + 1}").strip(),
                "tag": str(it.get("tag") or "公告").strip(),
                "level": level,
                "date": str(it.get("date") or datetime.now().strftime("%Y-%m-%d")).strip(),
                "pinned": bool(it.get("pinned", False)),
                "active": bool(it.get("active", True)),
                "content": str(it.get("content") or "").strip(),
            }
        )
    return out


def save_announcements_file(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    today = datetime.now().strftime("%Y-%m-%d")
    for idx, it in enumerate(items):
        content = str(it.get("content") or "").strip()
        if not content:
            continue
        level = str(it.get("level", "info")).strip().lower()
        if level not in ("info", "change", "warn"):
            level = "info"
        cleaned.append(
            {
                "id": str(it.get("id") or f"{today}-{idx + 1}").strip(),
                "tag": str(it.get("tag") or "公告").strip(),
                "level": level,
                "date": str(it.get("date") or today).strip(),
                "pinned": bool(it.get("pinned", False)),
                "active": bool(it.get("active", True)),
                "content": content,
            }
        )
    ANNOUNCEMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANNOUNCEMENTS_PATH.write_text(
        json.dumps({"items": cleaned}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return cleaned


def get_scene_edit_summary(db: CorpusDB) -> Dict[str, Any]:
    """统计人工锁定场景标签总量与今日修改场次数（用于公告自动填充与核对）."""
    now_local = datetime.now().astimezone()
    local_midnight = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    today_utc_iso = local_midnight.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with db.session() as conn:
        tot = conn.execute(
            """
            SELECT COUNT(*) AS parts, COUNT(DISTINCT bvid) AS bvids
              FROM videos
             WHERE scene_locked = 1
            """
        ).fetchone()
        tod = conn.execute(
            """
            SELECT COUNT(*) AS parts, COUNT(DISTINCT bvid) AS bvids
              FROM videos
             WHERE scene_locked = 1
               AND updated_at >= ?
            """,
            (today_utc_iso,),
        ).fetchone()
        by_sc = [
            dict(r)
            for r in conn.execute(
                """
                SELECT scene_type, COUNT(*) AS parts, COUNT(DISTINCT bvid) AS bvids
                  FROM videos
                 WHERE scene_locked = 1
                 GROUP BY scene_type
                 ORDER BY parts DESC
                """
            ).fetchall()
        ]
    return {
        "total_locked_parts": int(tot["parts"] or 0),
        "total_locked_bvids": int(tot["bvids"] or 0),
        "today_locked_parts": int(tod["parts"] or 0),
        "today_locked_bvids": int(tod["bvids"] or 0),
        "today_since_utc": today_utc_iso,
        "by_scene": by_sc,
    }


def get_asr_progress(db: CorpusDB) -> Dict[str, Any]:
    """统计本地无字幕待补录与旧版 ASR 高精度重录进度."""
    with db.session() as conn:
        nosub = conn.execute(
            """
            SELECT COUNT(*) AS parts, COALESCE(SUM(duration_sec), 0) AS dur
              FROM videos
             WHERE subtitle_status = 'no_subtitle' AND duration_sec >= 15
            """
        ).fetchone()
        legacy = conn.execute(
            """
            SELECT COUNT(*) AS parts, COALESCE(SUM(duration_sec), 0) AS dur
              FROM videos
             WHERE subtitle_status = 'ok' AND subtitle_lan = 'asr-large-v3-turbo'
            """
        ).fetchone()
        v2 = conn.execute(
            """
            SELECT COUNT(*) AS parts, COALESCE(SUM(duration_sec), 0) AS dur,
                   COALESCE(SUM(line_count), 0) AS lines
              FROM videos
             WHERE subtitle_status = 'ok' AND subtitle_lan = 'asr-v2-large-v3-turbo'
            """
        ).fetchone()
    legacy_parts = int(legacy["parts"] or 0)
    v2_parts = int(v2["parts"] or 0)
    total_retrans = legacy_parts + v2_parts
    legacy_dur = int(legacy["dur"] or 0)
    v2_dur = int(v2["dur"] or 0)
    total_dur = legacy_dur + v2_dur
    pct_parts = round((v2_parts / total_retrans) * 100, 1) if total_retrans > 0 else 100.0
    pct_dur = round((v2_dur / total_dur) * 100, 1) if total_dur > 0 else 100.0
    return {
        "no_subtitle_parts": int(nosub["parts"] or 0),
        "no_subtitle_hours": round(int(nosub["dur"] or 0) / 3600.0, 1),
        "legacy_asr_remaining": legacy_parts,
        "legacy_hours_remaining": round(legacy_dur / 3600.0, 1),
        "v2_asr_done": v2_parts,
        "v2_hours_done": round(v2_dur / 3600.0, 1),
        "v2_lines_total": int(v2["lines"] or 0),
        "retrans_total": total_retrans,
        "retrans_pct_parts": pct_parts,
        "retrans_pct_hours": pct_dur,
    }


def _parse_ssh_target(cloud_host: str) -> Tuple[str, str, int]:
    raw = (cloud_host or "").strip()
    if not raw:
        raise ValueError("请先填写云端服务器地址（例如 root@47.xx.xx.xx）")
    user = "root"
    host_part = raw
    if "@" in raw:
        user, host_part = raw.split("@", 1)
    port = 22
    if ":" in host_part:
        host_part, p_str = host_part.rsplit(":", 1)
        if p_str.isdigit():
            port = int(p_str)
    return user.strip(), host_part.strip(), port


def _open_ssh_client(cloud_host: str, password: Optional[str] = None):
    try:
        import paramiko
    except ImportError as exc:
        raise RuntimeError(
            "本地环境缺少 paramiko 依赖，请先执行 .\\.venv\\Scripts\\python.exe -m pip install paramiko"
        ) from exc

    user, hostname, port = _parse_ssh_target(cloud_host)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    key_path = find_local_ssh_key()
    pwd = (password or "").strip() or None
    connect_kwargs: Dict[str, Any] = {
        "hostname": hostname,
        "port": port,
        "username": user,
        "timeout": 15,
        "banner_timeout": 15,
        "auth_timeout": 15,
    }
    if pwd:
        connect_kwargs["password"] = pwd
        connect_kwargs["look_for_keys"] = True
        connect_kwargs["allow_agent"] = True
    elif key_path:
        connect_kwargs["key_filename"] = str(key_path)
        connect_kwargs["look_for_keys"] = True
        connect_kwargs["allow_agent"] = True
    else:
        raise RuntimeError("尚未配置 SSH 免密密钥，请在界面中输入服务器 SSH 密码（或点击「一键配置免密登录」）。")

    client.connect(**connect_kwargs)
    return client


def _exec_remote(client, cmd: str, *, timeout: int = 300) -> Tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def _fetch_public_stats(public_url: str) -> Optional[Dict[str, Any]]:
    if not public_url:
        return None
    try:
        r = httpx.get(
            f"{public_url.rstrip('/')}/api/v1/stats",
            headers={"Cache-Control": "no-cache"},
            timeout=12.0,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def _restart_remote_service(client, log_cb: LogCallback) -> None:
    cmd = (
        "SVC=$(systemctl list-unit-files --type=service --no-legend "
        "| awk '{print $1}' | grep -iE 'zhijiang|zjdb' | head -n 1 || true); "
        "if [ -n \"$SVC\" ]; then "
        "  echo \"restarting:$SVC\"; "
        "  systemctl restart \"$SVC\"; "
        "  sleep 1; "
        "  systemctl is-active \"$SVC\"; "
        "else "
        "  echo 'no_service_found'; "
        "fi"
    )
    code, out, err = _exec_remote(client, cmd, timeout=45)
    if code != 0:
        raise RuntimeError(f"重启云端服务失败: {err or out}")
    for line in out.strip().splitlines():
        log_cb(f"[云端服务] {line}")


def setup_ssh_key(cloud_host: str, password: str, log_cb: LogCallback) -> Dict[str, Any]:
    """一键在本地生成 ed25519 密钥并将公钥写入云端 authorized_keys，以后无需输密码."""
    if not (password or "").strip():
        raise ValueError("首次配置 SSH 免密登录需要输入一次服务器 SSH 密码。")

    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)
    priv_path = ssh_dir / "id_ed25519"
    pub_path = ssh_dir / "id_ed25519.pub"

    if not priv_path.exists() or not pub_path.exists():
        log_cb("正在本地生成 SSH Ed25519 密钥对 (~/.ssh/id_ed25519)...")
        ssh_keygen = shutil.which("ssh-keygen")
        if not ssh_keygen:
            raise RuntimeError("系统未找到 ssh-keygen 命令。")
        res = subprocess.run(
            [ssh_keygen, "-t", "ed25519", "-N", "", "-f", str(priv_path), "-C", "zhijiang-local-admin"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(f"生成 SSH 密钥失败: {res.stderr or res.stdout}")
    else:
        log_cb(f"检测到已有本机密钥: {priv_path}")

    pub_key = pub_path.read_text(encoding="utf-8").strip()
    if not pub_key:
        raise RuntimeError("读取本机公钥失败。")

    log_cb(f"正在连接 {cloud_host} 写入 ~/.ssh/authorized_keys ...")
    client = _open_ssh_client(cloud_host, password=password)
    try:
        safe_pub = pub_key.replace("'", "'\"'\"'")
        cmd = (
            "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
            "touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && "
            f"grep -qxF '{safe_pub}' ~/.ssh/authorized_keys || echo '{safe_pub}' >> ~/.ssh/authorized_keys"
        )
        code, out, err = _exec_remote(client, cmd, timeout=30)
        if code != 0:
            raise RuntimeError(f"写入云端 authorized_keys 失败: {err or out}")
    finally:
        client.close()

    log_cb("正在验证免密 SSH 登录...")
    verify_client = _open_ssh_client(cloud_host, password=None)
    try:
        code, out, _ = _exec_remote(verify_client, "echo ssh_key_ok", timeout=15)
        if code != 0 or "ssh_key_ok" not in out:
            raise RuntimeError("免密登录验证未通过，请检查服务器 sshd PubkeyAuthentication 配置。")
    finally:
        verify_client.close()

    log_cb("✅ SSH 免密登录配置成功！以后在本地后台同步云端不再需要输入密码。")
    return {"ok": True, "key_path": str(priv_path)}


def deploy_code_and_announcements(
    conf: Dict[str, Any],
    password: Optional[str],
    log_cb: LogCallback,
) -> Dict[str, Any]:
    """仅更新云端代码与站点公告，绝不触碰云端数据库."""
    cloud_host = conf["cloud_host"]
    app_dir = conf["cloud_app_dir"]
    pub_url = conf["public_url"]

    log_cb(f"正在连接云端 {cloud_host} ({app_dir})...")
    client = _open_ssh_client(cloud_host, password=password)
    try:
        log_cb("拉取 GitHub 最新代码 (git pull --ff-only origin main)...")
        cmd = (
            f"set -e; cd {app_dir}; "
            "git checkout -- config/announcements.json 2>/dev/null || true; "
            "git pull --ff-only origin main"
        )
        code, out, err = _exec_remote(client, cmd, timeout=60)
        if code != 0:
            raise RuntimeError(f"云端 git pull 失败: {err or out}")
        for ln in out.strip().splitlines():
            log_cb(f"[git] {ln}")

        if ANNOUNCEMENTS_PATH.exists():
            log_cb("通过 SFTP 同步本地 config/announcements.json 到云端...")
            sftp = client.open_sftp()
            try:
                remote_ann = posixpath.join(app_dir, "config", "announcements.json")
                sftp.put(str(ANNOUNCEMENTS_PATH), remote_ann)
            finally:
                sftp.close()

        _restart_remote_service(client, log_cb)
    finally:
        client.close()

    time.sleep(0.8)
    live = _fetch_public_stats(pub_url)
    if live:
        ann_n = len(live.get("announcements") or [])
        log_cb(f"✅ 线上验收成功：{pub_url} 当前生效公告 {ann_n} 条，收录跨度 {live.get('date_min')} ~ {live.get('date_max')}")
    return {"mode": "code_and_announcements", "live_stats": live}


def sync_scene_tags_to_cloud(
    db: CorpusDB,
    conf: Dict[str, Any],
    password: Optional[str],
    *,
    only_locked: bool = True,
    also_publish_announcements: bool = True,
    log_cb: LogCallback,
) -> Dict[str, Any]:
    """极速同步人工修正的场景标签到云端（仅数 KB，不触碰字幕正文与正在跑的 ASR）."""
    cloud_host = conf["cloud_host"]
    app_dir = conf["cloud_app_dir"]
    pub_url = conf["public_url"]

    TMP_SYNC_DIR.mkdir(parents=True, exist_ok=True)
    patch_file = TMP_SYNC_DIR / "scene_tags_patch.json.gz"
    exp = export_scene_tags(db, patch_file, only_locked=only_locked)
    log_cb(
        f"已导出纯场景标签补丁：{exp['parts_exported']} 个分P（{exp['bvids_exported']} 场直播），"
        f"补丁体积仅 {exp['patch_size_kb']} KB"
    )

    client = _open_ssh_client(cloud_host, password=password)
    try:
        log_cb("确保云端代码最新（含 apply-scene-tags 命令）...")
        code, out, err = _exec_remote(
            client,
            f"set -e; cd {app_dir}; git checkout -- config/announcements.json 2>/dev/null || true; git pull --ff-only origin main",
            timeout=60,
        )
        if code != 0:
            raise RuntimeError(f"云端 git pull 失败: {err or out}")

        sftp = client.open_sftp()
        try:
            remote_patch = "/tmp/zhijiang_scene_tags.json.gz"
            log_cb("正在通过 SFTP 上传轻量标签补丁...")
            sftp.put(str(patch_file), remote_patch)
            if also_publish_announcements and ANNOUNCEMENTS_PATH.exists():
                log_cb("同步本地 config/announcements.json 到云端...")
                sftp.put(str(ANNOUNCEMENTS_PATH), posixpath.join(app_dir, "config", "announcements.json"))
        finally:
            sftp.close()

        log_cb("正在云端热更新场景标签（零停服、不动字幕文本）...")
        code, out, err = _exec_remote(
            client,
            f"set -e; cd {app_dir}; .venv/bin/python run.py apply-scene-tags /tmp/zhijiang_scene_tags.json.gz && rm -f /tmp/zhijiang_scene_tags.json.gz",
            timeout=120,
        )
        if code != 0:
            raise RuntimeError(f"云端应用标签补丁失败: {err or out}")
        apply_info = json.loads(out.strip())
        log_cb(
            f"✅ 云端标签热更新完成：实际变动 {apply_info.get('parts_updated', 0)} 个分P"
            f"（{apply_info.get('bvids_updated', 0)} 场直播），联动更新 {apply_info.get('subtitles_updated', 0):,} 条字幕标签。"
        )
    finally:
        client.close()
        patch_file.unlink(missing_ok=True)

    live = _fetch_public_stats(pub_url)
    return {
        "mode": "scene_tags_only",
        "exported": exp,
        "applied": apply_info,
        "live_stats": live,
    }


def sync_database_to_cloud(
    db: CorpusDB,
    conf: Dict[str, Any],
    password: Optional[str],
    *,
    run_ingest: bool = False,
    run_asr: bool = False,
    ingest_limit: int = 10,
    also_publish_announcements: bool = True,
    log_cb: LogCallback,
) -> Dict[str, Any]:
    """完整一键更新或纯差异增量同步本地数据库到云端（零停服 WAL 热写入）."""
    cloud_host = conf["cloud_host"]
    app_dir = conf["cloud_app_dir"]
    pub_url = conf["public_url"]

    before_stats = _fetch_public_stats(pub_url)
    if before_stats:
        log_cb(
            f"[同步前公网] 字幕 {before_stats.get('subtitles_total', 0):,} 条 | "
            f"分P {before_stats.get('video_parts_ok', 0):,} | "
            f"跨度 {before_stats.get('date_min')} ~ {before_stats.get('date_max')}"
        )

    if run_ingest:
        cfg = load_config(db_path=db.db_path)
        log_cb(f"==> [1/4] 增量抓取 6 位成员最新直播回放（每人检查最近 {ingest_limit} 场）...")
        for m in cfg.members:
            s = ingest_from_up(
                db,
                cfg,
                m.up_name,
                limit=ingest_limit,
                source="auto",
                progress_cb=log_cb,
            )
            if s.parts_ok > 0:
                log_cb(f"  [{m.up_name}] 新增 {s.parts_ok} 个分P / {s.lines_inserted:,} 条字幕")

        if run_asr:
            from .asr_backfill import run_asr_backfill

            log_cb("==> [2/4] 检查无字幕新分P，调用本地 GPU faster-whisper 自动补录...")
            os.environ["HF_HOME"] = str(PROJECT_ROOT / "data" / "cache" / "huggingface")
            os.environ["TMP"] = str(TMP_SYNC_DIR)
            os.environ["TEMP"] = str(TMP_SYNC_DIR)
            run_asr_backfill(
                db,
                cfg,
                limit=0,
                batch_size=12,
                beam_size=2,
                chunk_length=10,
                word_timestamps=True,
                workers=1,
                prefetch=1,
                progress_cb=log_cb,
            )
        else:
            log_cb("==> [2/4] 按选项跳过本地 GPU 补录")

        log_cb("==> [3/4] 执行场景初筛（仅处理未标注场次）...")
        tagged = run_scene_tagging(db, cfg, only_unknown=True, dry_run=False)
        log_cb(f"  场景初筛完成：处理 {len(tagged)} 个分P。")

    log_cb("==> [4/4] 拉取云端数据清单并计算增量差异补丁...")
    TMP_SYNC_DIR.mkdir(parents=True, exist_ok=True)
    delta_file = TMP_SYNC_DIR / "web_admin_delta.json.gz"
    delta_file.unlink(missing_ok=True)

    client = _open_ssh_client(cloud_host, password=password)
    try:
        code, out, err = _exec_remote(
            client,
            f"set -e; cd {app_dir}; git checkout -- config/announcements.json 2>/dev/null || true; "
            "git pull --ff-only -q origin main; .venv/bin/python run.py sync-manifest",
            timeout=120,
        )
        if code != 0:
            raise RuntimeError(f"拉取云端清单失败: {err or out}")
        remote_manifest = json.loads(out.strip()) if out.strip() else {}

        exp = export_delta(db, remote_manifest, delta_file)
        log_cb(
            f"差异比对完成：本地 {exp['local_parts']} / 云端 {exp['remote_parts']} | "
            f"需同步 {exp['parts_changed']} 个分P（新增 {exp['parts_new']}，更新 {exp['parts_updated']}）| "
            f"字幕 {exp['subtitles_exported']:,} 条 | 补丁 {exp['delta_size_kb']} KB"
        )

        sftp = client.open_sftp()
        try:
            if also_publish_announcements and ANNOUNCEMENTS_PATH.exists():
                log_cb("同步本地 config/announcements.json 到云端...")
                sftp.put(str(ANNOUNCEMENTS_PATH), posixpath.join(app_dir, "config", "announcements.json"))

            apply_info: Dict[str, Any] = {"parts_applied": 0, "subtitles_inserted": 0}
            if int(exp["parts_changed"]) > 0:
                remote_delta = "/tmp/zhijiang_delta.json.gz"
                log_cb(f"正在通过 SFTP 上传增量补丁 ({exp['delta_size_kb']} KB)...")
                sftp.put(str(delta_file), remote_delta)
        finally:
            sftp.close()

        if int(exp["parts_changed"]) > 0:
            log_cb("正在云端热合并入库并增量更新 FTS5 索引（零停服）...")
            code, out, err = _exec_remote(
                client,
                f"set -e; cd {app_dir}; .venv/bin/python run.py apply-delta /tmp/zhijiang_delta.json.gz && rm -f /tmp/zhijiang_delta.json.gz",
                timeout=600,
            )
            if code != 0:
                raise RuntimeError(f"云端热合并失败: {err or out}")
            apply_info = json.loads(out.strip())
            log_cb(
                f"✅ 增量热写入完成：{apply_info.get('parts_applied', 0)} 个分P / "
                f"{apply_info.get('subtitles_inserted', 0):,} 条字幕。"
            )
        else:
            log_cb("✅ 云端数据库已与本地 100% 一致，无需上传字幕补丁。")
    finally:
        client.close()
        delta_file.unlink(missing_ok=True)

    after_stats = _fetch_public_stats(pub_url)
    if after_stats:
        d_sub = (
            after_stats.get("subtitles_total", 0) - before_stats.get("subtitles_total", 0)
            if before_stats
            else apply_info.get("subtitles_inserted", 0)
        )
        lv = after_stats.get("latest_video") or {}
        log_cb(
            f"[同步后公网] 字幕 {after_stats.get('subtitles_total', 0):,} 条（{d_sub:+,}）| "
            f"分P {after_stats.get('video_parts_ok', 0):,} | "
            f"最新录播: [{lv.get('up_name', '')}] {lv.get('title', '')}"
        )

    return {
        "mode": "full_sync" if run_ingest else "db_sync_only",
        "exported": exp,
        "applied": apply_info,
        "live_stats": after_stats,
    }


def backfill_cloud_metrics(
    conf: Dict[str, Any],
    password: Optional[str],
    log_cb: LogCallback,
) -> Dict[str, Any]:
    """从云端 Nginx 历史访问日志（/var/log/nginx/access.log*）幂等补回上线以来的真实访客数."""
    cloud_host = conf["cloud_host"]
    app_dir = conf["cloud_app_dir"]
    pub_url = conf["public_url"]

    log_cb(f"正在连接云端 {cloud_host} 拉取最新代码并解析 /var/log/nginx/access.log* ...")
    client = _open_ssh_client(cloud_host, password=password)
    try:
        code, out, err = _exec_remote(
            client,
            f"set -e; cd {app_dir}; git checkout -- config/announcements.json 2>/dev/null || true; "
            "git pull --ff-only -q origin main; .venv/bin/python run.py backfill-metrics --log-dir /var/log/nginx",
            timeout=120,
        )
        if code != 0:
            raise RuntimeError(f"云端回灌历史访问统计失败: {err or out}")
        info = json.loads(out.strip())
        v = info.get("visitors") or {}
        log_cb(
            f"✅ 扫描 {len(info.get('files_scanned') or [])} 个 Nginx 日志文件（{info.get('lines_scanned', 0):,} 行），"
            f"命中真实首页访问 {info.get('hits_matched', 0):,} 次。"
        )
        log_cb(
            f"✅ 历史访问统计已补齐：累计访客 {v.get('total_uv', 0):,} 人（今日 {v.get('today_uv', 0):,}），"
            f"累计访问 {v.get('total_pv', 0):,} 次！"
        )
    finally:
        client.close()

    live = _fetch_public_stats(pub_url)
    return {"mode": "backfill_metrics", "backfill": info, "live_stats": live}
