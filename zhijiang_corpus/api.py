"""兼容转发层：后端 FastAPI 应用已独立迁移至 backend/app.py."""

import backend.app as _backend_app
from backend.app import create_app

__all__ = ["app", "create_app", "local_app", "public_app"]


def __getattr__(name: str):
    if name in ("app", "local_app", "public_app"):
        return getattr(_backend_app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
