"""SQLite checkpointer: persists graph state after every step so paused or
interrupted runs (human approval, quota exhaustion, crashes) can resume."""

from __future__ import annotations

import enum
import inspect
import sqlite3
from pathlib import Path
from types import ModuleType

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from app.models import agent_outputs, domain

STATE_MODEL_MODULES: tuple[ModuleType, ...] = (domain, agent_outputs)


def allowed_state_types() -> list[tuple[str, str]]:
    """Explicit allow-list of our Pydantic/Enum types for checkpoint deserialisation
    (LangGraph is moving to block unregistered types)."""
    allowed = []
    for module in STATE_MODEL_MODULES:
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__ and issubclass(obj, (BaseModel, enum.Enum)):
                allowed.append((module.__name__, name))
    return allowed


def create_checkpointer(path: Path | str) -> SqliteSaver:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(connection, serde=JsonPlusSerializer(allowed_msgpack_modules=allowed_state_types()))
    saver.setup()
    return saver
