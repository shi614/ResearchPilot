from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import PROJECT_ROOT, Settings


def test_defaults_use_free_tier_flash_and_low_quota_limits() -> None:
    settings = Settings(_env_file=None)
    assert "flash" in settings.gemini_model
    assert settings.max_revisions == 1
    assert settings.max_research_tool_rounds == 2


def test_relative_paths_resolve_against_project_root() -> None:
    settings = Settings(_env_file=None, chroma_dir=Path("data/chroma"))
    assert settings.chroma_dir == (PROJECT_ROOT / "data" / "chroma").resolve()
    assert settings.database_url == (
        f"sqlite:///{(PROJECT_ROOT / 'data' / 'researchpilot.db').resolve().as_posix()}"
    )


def test_non_sqlite_database_url_is_left_untouched() -> None:
    url = "postgresql+psycopg://user:pw@localhost/researchpilot"
    assert Settings(_env_file=None, database_url=url).database_url == url


def test_missing_keys_are_reported_and_blank_keys_treated_as_missing() -> None:
    settings = Settings(_env_file=None, gemini_api_key="  ", tavily_api_key="tvly-abc")
    assert settings.gemini_api_key is None
    assert settings.missing_required_keys() == ["GEMINI_API_KEY"]


def test_api_keys_never_appear_in_repr_or_dump() -> None:
    secret = "AIza-super-secret-value"
    settings = Settings(_env_file=None, gemini_api_key=secret)
    assert secret not in repr(settings)
    assert secret not in settings.model_dump_json()
    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == secret


def test_models_prefix_is_stripped() -> None:
    assert Settings(_env_file=None, gemini_model="models/gemini-2.5-flash").gemini_model == (
        "gemini-2.5-flash"
    )


def test_env_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.setenv("MAX_REVISIONS", "2")
    settings = Settings(_env_file=None)
    assert settings.gemini_model == "gemini-2.5-flash-lite"
    assert settings.max_revisions == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_revisions", 10), ("gemini_temperature", 5), ("gemini_max_rpm", 0), ("gemini_model", " ")],
)
def test_invalid_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_ensure_directories_creates_runtime_folders(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'db' / 'x.db').as_posix()}",
        chroma_dir=tmp_path / "chroma",
        upload_dir=tmp_path / "uploads",
        reports_dir=tmp_path / "reports",
    )
    settings.ensure_directories()
    for directory in ("db", "chroma", "uploads", "reports"):
        assert (tmp_path / directory).is_dir()
