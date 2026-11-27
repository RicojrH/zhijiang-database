"""FastAPI 后端主入口：严格区分「公网只读部署模式」与「本地运维调试模式」.

启动方式：
  - 本地开发与抓取运维（挂载 /api/v1/* + /api/v1/admin/* + /debug）：
      python run.py serve
      或 uvicorn backend.app:local_app --host 127.0.0.1 --port 8000

  - 云端公网只读部署（仅挂载 /api/v1/*，零写操作、零管理接口、零调试页面）：
      uvicorn backend.app:public_app --host 0.0.0.0 --port 8000
      或设置环境变量 ZJ_ENABLE_ADMIN=0 后启动 uvicorn backend.app:app
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from zhijiang_corpus.config import PROJECT_ROOT, load_config
from zhijiang_corpus.db import CorpusDB

from .routers.public import build_public_router

DEBUG_CONSOLE_PATH = Path(__file__).resolve().parent / "debug_console.html"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("ZJ_CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app(
    db_path: Optional[str | Path] = None,
    *,
    enable_admin: Optional[bool] = None,
) -> FastAPI:
    if enable_admin is None:
        enable_admin = os.getenv("ZJ_ENABLE_ADMIN", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )

    cfg = load_config(db_path=db_path)
    db = CorpusDB(cfg.db_path)
    db.init_db()

    app = FastAPI(
        title="枝江直播回放 AI 字幕语料库 - Backend API",
        version="0.3.0",
        description=(
            "前后端分离架构后端服务。"
            "公网部署时请使用 public_app（仅暴露 /api/v1/* 只读检索接口）；"
            "本地数据入库运维时启用 admin 路由（/api/v1/admin/*）与 /debug 调试台。"
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "OPTIONS"] if enable_admin else ["GET", "OPTIONS"],
        allow_headers=["*"],
    )

    # 1. 公网只读检索 API（始终挂载）
    app.include_router(build_public_router(db, cfg))

    # 2. 本地私有管理 API 与运维调试台（仅本地模式挂载，云端部署默认彻底关闭且不加载 admin 依赖）
    if enable_admin:
        from .routers.admin import build_admin_router

        app.include_router(build_admin_router(db))

        @app.get("/debug", response_class=HTMLResponse, include_in_schema=False)
        @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
        def debug_console() -> str:
            return DEBUG_CONSOLE_PATH.read_text(encoding="utf-8")

    # 3. 公开前端静态站点挂载（单机部署时 / 直接提供面向公众的前端；若前后端分域部署，可设 ZJ_SERVE_FRONTEND=0）
    serve_frontend = os.getenv("ZJ_SERVE_FRONTEND", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    index_html = FRONTEND_DIR / "index.html"
    if serve_frontend and index_html.exists():
        @app.get("/", include_in_schema=False)
        def public_frontend() -> FileResponse:
            return FileResponse(str(index_html))

        for asset_name in ("styles.css", "app.js", "config.js"):
            asset_file = FRONTEND_DIR / asset_name
            if asset_file.exists():
                def _make_asset_handler(path_str: str):
                    def _serve_asset() -> FileResponse:
                        return FileResponse(path_str)
                    return _serve_asset

                app.add_api_route(
                    f"/{asset_name}",
                    _make_asset_handler(str(asset_file)),
                    methods=["GET"],
                    include_in_schema=False,
                )

        app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend_assets")
    else:
        @app.get("/", include_in_schema=False)
        def public_root() -> dict[str, str]:
            return {
                "service": "zhijiang-subtitle-backend",
                "mode": "public_read_only" if not enable_admin else "local_admin",
                "docs": "/docs",
                "health": "/api/v1/health",
            }

    return app


# 供 Uvicorn 按需懒加载的三种明确实例（避免云端加载 public_app 时触发 local_app/admin 依赖）：
# - backend.app:public_app -> 云端生产只读后端（推荐上云使用）
# - backend.app:local_app  -> 本地含运维与调试台的完整实例
# - backend.app:app        -> 按环境变量 ZJ_ENABLE_ADMIN 自动切换
def __getattr__(name: str) -> FastAPI:
    if name == "public_app":
        inst = create_app(enable_admin=False)
        globals()[name] = inst
        return inst
    if name == "local_app":
        inst = create_app(enable_admin=True)
        globals()[name] = inst
        return inst
    if name == "app":
        inst = create_app()
        globals()[name] = inst
        return inst
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
