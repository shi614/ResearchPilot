"""ResearchPilot launcher.

Usage:
    python run.py all [--reload]              # backend + Streamlit UI together
    python run.py backend [--reload]          # FastAPI on API_HOST:API_PORT (default 127.0.0.1:8000)
    python run.py frontend [--port 8501]      # Streamlit UI (expects the backend to be running)
    python run.py research "question" [...]   # run the multi-agent workflow in the terminal
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_APP = PROJECT_ROOT / "frontend" / "streamlit_app.py"
sys.path.insert(0, str(BACKEND_DIR))


def run_backend(reload: bool) -> None:
    import uvicorn

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


def frontend_command(port: int) -> list[str]:
    return [sys.executable, "-m", "streamlit", "run", str(FRONTEND_APP), "--server.port", str(port)]


def run_frontend(port: int) -> None:
    raise SystemExit(subprocess.call(frontend_command(port), cwd=PROJECT_ROOT))


def _wait_for_backend(url: str, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def run_all(reload: bool, port: int) -> None:
    from app.config import get_settings

    backend_cmd = [sys.executable, str(Path(__file__).resolve()), "backend"] + (["--reload"] if reload else [])
    backend = subprocess.Popen(backend_cmd, cwd=PROJECT_ROOT)
    try:
        if not _wait_for_backend(get_settings().backend_url):
            raise SystemExit("The backend did not start; see the log above.")
        print(f"\nResearchPilot UI: http://localhost:{port}\n", flush=True)
        subprocess.call(frontend_command(port), cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        pass
    finally:
        backend.terminate()
        backend.wait(timeout=10)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "research":
        from app.cli import main as research_cli

        research_cli(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description="Run ResearchPilot services.")
    parser.add_argument("service", choices=["all", "backend", "frontend", "research"], help="What to start")
    parser.add_argument("--reload", action="store_true", help="Auto-reload the backend on code changes")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit port (frontend/all)")
    args = parser.parse_args()
    if args.service == "backend":
        run_backend(reload=args.reload)
    elif args.service == "frontend":
        run_frontend(args.port)
    elif args.service == "all":
        run_all(args.reload, args.port)


if __name__ == "__main__":
    main()
