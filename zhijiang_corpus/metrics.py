"""站点轻量访问统计模块（独立 SQLite 文件 site_metrics.db，零锁竞争，隐私友好）.

设计原则：
  - 独立存放于 data/site_metrics.db（与 2GB 主语料库完全隔离，不影响检索与增量热同步）。
  - 仅保存单向截断哈希，绝不存储访客原始 IP。
  - 按北京时间（UTC+8）切分「今日访客 / 今日访问」，自动过滤常见爬虫与探活脚本。
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

_CST = timezone(timedelta(hours=8))
_BOT_UA_RE = re.compile(
    r"(?:bot|spider|crawl|slurp|curl/|wget/|python-httpx|python-requests|"
    r"headless|lighthouse|pingdom|uptimerobot|statuscake|monitor|scrapy|go-http-client)",
    re.IGNORECASE,
)


def _today_cst() -> str:
    return datetime.now(_CST).strftime("%Y-%m-%d")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SiteMetricsDB:
    """轻量独立访客（UV）与访问量（PV）计数器."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS daily_pv (
                    visit_date TEXT PRIMARY KEY,
                    pv INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS daily_visitors (
                    visit_date TEXT NOT NULL,
                    visitor_hash TEXT NOT NULL,
                    PRIMARY KEY (visit_date, visitor_hash)
                );
                CREATE TABLE IF NOT EXISTS all_visitors (
                    visitor_hash TEXT PRIMARY KEY,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _hash_visitor(client_ip: str, user_agent: str) -> str:
        ip = (client_ip or "unknown").strip()
        ua = (user_agent or "").strip()[:120]
        raw = f"zjdb-v1|{ip}|{ua}".encode("utf-8", errors="ignore")
        return hashlib.sha256(raw).hexdigest()[:20]

    @staticmethod
    def is_bot(user_agent: str) -> bool:
        ua = (user_agent or "").strip()
        if not ua:
            return False
        return bool(_BOT_UA_RE.search(ua))

    def _query_summary(self, conn: sqlite3.Connection, today: str) -> Dict[str, int]:
        total_uv = int(conn.execute("SELECT COUNT(*) FROM all_visitors").fetchone()[0] or 0)
        total_pv = int(conn.execute("SELECT COALESCE(SUM(pv), 0) FROM daily_pv").fetchone()[0] or 0)
        today_uv = int(
            conn.execute(
                "SELECT COUNT(*) FROM daily_visitors WHERE visit_date = ?",
                (today,),
            ).fetchone()[0]
            or 0
        )
        row_pv = conn.execute(
            "SELECT pv FROM daily_pv WHERE visit_date = ?",
            (today,),
        ).fetchone()
        today_pv = int(row_pv[0] or 0) if row_pv else 0
        return {
            "total_uv": total_uv,
            "total_pv": total_pv,
            "today_uv": today_uv,
            "today_pv": today_pv,
        }

    def get_summary(self) -> Dict[str, int]:
        today = _today_cst()
        conn = self._connect()
        try:
            return self._query_summary(conn, today)
        finally:
            conn.close()

    def record_visit(self, client_ip: str, user_agent: str) -> Dict[str, int]:
        today = _today_cst()
        conn = self._connect()
        try:
            if not self.is_bot(user_agent):
                vhash = self._hash_visitor(client_ip, user_agent)
                now = _now_utc()
                conn.execute(
                    """
                    INSERT INTO daily_pv (visit_date, pv) VALUES (?, 1)
                    ON CONFLICT(visit_date) DO UPDATE SET pv = pv + 1
                    """,
                    (today,),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO daily_visitors (visit_date, visitor_hash) VALUES (?, ?)",
                    (today, vhash),
                )
                conn.execute(
                    """
                    INSERT INTO all_visitors (visitor_hash, first_seen, last_seen)
                    VALUES (?, ?, ?)
                    ON CONFLICT(visitor_hash) DO UPDATE SET last_seen = excluded.last_seen
                    """,
                    (vhash, now, now),
                )
                conn.commit()
            return self._query_summary(conn, today)
        finally:
            conn.close()
