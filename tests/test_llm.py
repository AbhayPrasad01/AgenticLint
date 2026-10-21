from __future__ import annotations

import pytest
from pydantic import BaseModel

from review_agent.llm import AgentLoopError, run_tool_loop
from tests.fakes import OneShotClient, tool_use


class Echo(BaseModel):
    message: str


SYSTEM = "You are a test assistant."


def test_successful_submit_on_first_turn():
    client = OneShotClient([tool_use("submit_result", {"message": "hello"})], stop_reason="tool_use")
    result, usage = run_tool_loop(system=SYSTEM, prompt="go", output_model=Echo, client=client)
    assert result.message == "hello"
    assert usage["llm_calls"] == 1


def test_end_turn_raises_immediately():
    """Model returns end_turn with no tools -> raise immediately, no nudge."""
    client = OneShotClient([], stop_reason="end_turn")
    with pytest.raises(AgentLoopError, match="end_turn"):
        run_tool_loop(system=SYSTEM, prompt="go", output_model=Echo, client=client, max_turns=8)


def test_max_tokens_raises_immediately():
    """Model hits max_tokens with no tools -> raise immediately."""
    client = OneShotClient([], stop_reason="max_tokens")
    with pytest.raises(AgentLoopError, match="max_tokens"):
        run_tool_loop(system=SYSTEM, prompt="go", output_model=Echo, client=client)


def test_unknown_tool_returns_error_to_model_and_retries():
    """Calling an unknown tool sends is_error feedback; second turn submits correctly."""
    from types import SimpleNamespace

    calls = []

    def scripted_create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return SimpleNamespace(
                content=[tool_use("nonexistent_tool", {})],
                stop_reason="tool_use",
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            )
        return SimpleNamespace(
            content=[tool_use("submit_result", {"message": "recovered"})],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )

    fake = SimpleNamespace(messages=SimpleNamespace(create=scripted_create))
    result, usage = run_tool_loop(system=SYSTEM, prompt="go", output_model=Echo, client=fake, max_turns=4)
    assert result.message == "recovered"
    assert usage["llm_calls"] == 2
    # The message history for the second LLM call is:
    #   [user(initial), assistant(turn1), user(tool_results)]
    second_call_messages = calls[1]["messages"]
    # Find the user message that carries tool results (role=user, content is a list of dicts)
    tool_result_msgs = [m for m in second_call_messages
                        if m["role"] == "user" and isinstance(m["content"], list)]
    assert tool_result_msgs, "Expected a user message with tool results in the second call"
    error_blocks = [b for b in tool_result_msgs[-1]["content"]
                    if isinstance(b, dict) and b.get("is_error")]
    assert len(error_blocks) == 1
    assert "nonexistent_tool" in error_blocks[0]["content"]
