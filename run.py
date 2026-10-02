"""ResearchPilot launcher.

Usage:
    python run.py backend      # FastAPI on API_HOST:API_PORT (default 127.0.0.1:8000)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"


def run_backend(reload: bool) -> None:
    import uvicorn

    sys.path.insert(0, str(BACKEND_DIR))
    from app.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=reload,
        reload_dirs=[str(BACKEND_DIR)] if reload else None,
        app_dir=str(BACKEND_DIR),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ResearchPilot services.")
    parser.add_argument("service", choices=["backend"], help="Service to start")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()
    if args.service == "backend":
        run_backend(reload=args.reload)


if __name__ == "__main__":
    main()
