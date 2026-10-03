"""GeminiLLM gateway: call counting, retry policy, malformed-output repair."""

from __future__ import annotations

from typing import Any

import pytest
from google.genai import errors as genai_errors
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from app.agents.llm import GeminiLLM, create_llm, should_retry_llm_error
from app.config import Settings
from app.exceptions import ConfigurationError, MalformedResponseError, RateLimitError


class Answer(BaseModel):
    text: str


class FakeRunnable:
    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = outputs
        self.inputs: list[Any] = []

    def invoke(self, messages: Any) -> Any:
        self.inputs.append(messages)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


class FakeChat:
    def __init__(self, outputs: list[Any]) -> None:
        self.runnable = FakeRunnable(outputs)
        self.structured_kwargs: dict[str, Any] = {}

    def with_structured_output(self, schema: type, **kwargs: Any) -> FakeRunnable:
        self.structured_kwargs = kwargs
        return self.runnable

    def bind_tools(self, tools: list) -> FakeRunnable:
        return self.runnable


def quota(per_day: bool = False, retry_delay: str | None = None) -> genai_errors.ClientError:
    quota_id = "GenerateRequestsPerDayPerProjectPerModel-FreeTier" if per_day else "GenerateRequestsPerMinute"
    details: list[dict] = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                            "violations": [{"quotaId": quota_id}]}]
    if retry_delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    return genai_errors.ClientError(429, {"error": {"code": 429, "message": "quota", "details": details}})


def gateway(outputs: list[Any]) -> tuple[GeminiLLM, FakeChat, list[float]]:
    chat = FakeChat(outputs)
    sleeps: list[float] = []
    return GeminiLLM(chat, max_rpm=6000, sleep=sleeps.append), chat, sleeps  # type: ignore[arg-type]


def gateway_pool(outputs_per_key: list[list[Any]]) -> tuple[GeminiLLM, list[FakeChat], list[float]]:
    """Like `gateway`, but backed by a pool of keys (one FakeChat per key)."""
    chats = [FakeChat(outputs) for outputs in outputs_per_key]
    sleeps: list[float] = []
    llm = GeminiLLM(chats, max_rpm=6000, sleep=sleeps.append)  # type: ignore[arg-type]
    return llm, chats, sleeps


def test_structured_output_uses_json_schema_and_counts_calls() -> None:
    llm, chat, _ = gateway([{"parsed": Answer(text="hi"), "parsing_error": None}])
    assert llm.structured(Answer, "sys", "user", operation="test").text == "hi"
    assert chat.structured_kwargs == {"method": "json_schema", "include_raw": True}
    assert llm.call_count == 1


def test_malformed_output_gets_one_repair_attempt() -> None:
    llm, chat, _ = gateway([
        {"parsed": None, "parsing_error": ValueError("missing field text")},
        {"parsed": Answer(text="fixed"), "parsing_error": None},
    ])
    assert llm.structured(Answer, "sys", "user", operation="test").text == "fixed"
    assert llm.call_count == 2
    assert "did not match the required JSON schema" in chat.runnable.inputs[1][-1].content


def test_malformed_output_twice_raises() -> None:
    bad = {"parsed": None, "parsing_error": ValueError("bad")}
    llm, _, _ = gateway([bad, bad])
    with pytest.raises(MalformedResponseError):
        llm.structured(Answer, "sys", "user", operation="test")


def test_daily_quota_fails_fast_without_retry() -> None:
    llm, _, sleeps = gateway([quota(per_day=True, retry_delay="20s")])
    with pytest.raises(RateLimitError, match="DAILY"):
        llm.structured(Answer, "sys", "user", operation="test")
    assert llm.call_count == 1 and sleeps == []


def test_short_per_minute_limit_is_retried_once() -> None:
    llm, _, sleeps = gateway([quota(retry_delay="12s"), {"parsed": Answer(text="ok"), "parsing_error": None}])
    assert llm.structured(Answer, "sys", "user", operation="test").text == "ok"
    assert sleeps == [12.0] and llm.call_count == 2


def test_rate_limit_without_retry_hint_is_not_retried() -> None:
    llm, _, sleeps = gateway([quota()])
    with pytest.raises(RateLimitError):
        llm.structured(Answer, "sys", "user", operation="test")
    assert sleeps == []


def test_retry_policy_classification() -> None:
    assert should_retry_llm_error(genai_errors.ServerError(503, {"error": {"code": 503}}))
    assert not should_retry_llm_error(quota(retry_delay="300s"))  # too long to wait
    assert not should_retry_llm_error(genai_errors.ClientError(400, {"error": {"code": 400}}))


def test_invoke_with_tools_returns_ai_message() -> None:
    message = AIMessage(content="", tool_calls=[{"name": "tavily_search", "args": {"query": "q"}, "id": "1"}])
    llm, _, _ = gateway([message])
    assert llm.invoke_with_tools([], [], operation="tools").tool_calls[0]["name"] == "tavily_search"


def test_create_llm_requires_key_and_disables_sdk_retries() -> None:
    with pytest.raises(ConfigurationError):
        create_llm(Settings(_env_file=None))
    llm = create_llm(Settings(_env_file=None, gemini_api_key="test-key", gemini_model="gemini-3.5-flash"))
    chat = llm._chat  # noqa: SLF001
    assert chat.model.endswith("gemini-3.5-flash") and chat.max_retries == 1  # type: ignore[attr-defined]


def test_create_llm_builds_one_chat_per_rotation_key() -> None:
    llm = create_llm(
        Settings(_env_file=None, gemini_api_keys="key-a,key-b,key-c", gemini_model="gemini-3.5-flash")
    )
    assert len(llm._chats) == 3  # noqa: SLF001
    assert llm.active_key_index == 0


def test_daily_quota_rotates_to_next_key_and_succeeds() -> None:
    llm, chats, sleeps = gateway_pool([
        [quota(per_day=True, retry_delay="20s")],
        [{"parsed": Answer(text="from key 2"), "parsing_error": None}],
    ])
    assert llm.structured(Answer, "sys", "user", operation="test").text == "from key 2"
    # One failed attempt on key 1, one successful attempt on key 2; no sleeping -
    # a daily-quota 429 rotates instead of waiting.
    assert llm.call_count == 2 and sleeps == []
    assert llm.active_key_index == 1
    assert chats[1].runnable.inputs  # the second key actually received the request


def test_daily_quota_raises_only_after_every_key_is_exhausted() -> None:
    exhausted = quota(per_day=True, retry_delay="20s")
    llm, _, sleeps = gateway_pool([[exhausted], [exhausted], [exhausted]])
    with pytest.raises(RateLimitError, match="DAILY"):
        llm.structured(Answer, "sys", "user", operation="test")
    assert llm.call_count == 3 and sleeps == []
    assert llm.active_key_index == 2


def test_single_key_daily_quota_behaviour_is_unchanged_by_rotation_support() -> None:
    """A single-key deployment (the common case) must still fail fast, exactly as before."""
    llm, chat, sleeps = gateway([quota(per_day=True, retry_delay="20s")])
    with pytest.raises(RateLimitError, match="DAILY"):
        llm.structured(Answer, "sys", "user", operation="test")
    assert llm.call_count == 1 and sleeps == []
    assert llm.active_key_index == 0
    assert chat is llm._chat  # noqa: SLF001
