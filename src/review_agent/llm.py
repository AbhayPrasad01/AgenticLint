"""A small, explicit Claude tool-calling loop.

Each agent gets a set of tools plus a special `submit_result` tool whose input
schema is the agent's Pydantic output model. `tool_choice={"type": "any"}`
forces Claude to call *some* tool every turn, so the loop always ends with a
validated, typed result (or an error that LangGraph's retry policy catches).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

DEFAULT_MODEL = os.environ.get("REVIEW_MODEL", "claude-sonnet-4-5")
SUBMIT_TOOL = "submit_result"
MAX_TOOL_RESULT_CHARS = 20_000

# USD per million tokens (input, output). Verify against https://docs.claude.com pricing before
# publishing cost numbers, or override with REVIEW_PRICE_PER_MTOK="input,output".
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-5-20250514": (3.0, 15.0),
}

T = TypeVar("T", bound=BaseModel)


class AgentLoopError(RuntimeError):
    """The model did not produce a valid result within the turn budget."""


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., Any]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


def empty_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_creation_tokens": 0, "llm_calls": 0}


def add_usage(a: dict | None, b: dict | None) -> dict:
    """LangGraph reducer: sum token usage coming from parallel nodes."""
    out = empty_usage()
    for d in (a or {}, b or {}):
        for k in out:
            out[k] += d.get(k, 0)
    return out


def price_for(model: str) -> tuple[float, float] | None:
    override = os.environ.get("REVIEW_PRICE_PER_MTOK")
    if override:
        inp, out = (float(x) for x in override.split(","))
        return inp, out
    return PRICES_PER_MTOK.get(model)


def cost_usd(usage: dict, model: str) -> float | None:
    price = price_for(model)
    if price is None:
        return None
    # cache_read tokens are billed at 10% of input price; cache_creation at 25% extra
    billed_input = (
        usage["input_tokens"] / 1e6 * price[0]
        + usage.get("cache_read_tokens", 0) / 1e6 * price[0] * 0.1
        + usage.get("cache_creation_tokens", 0) / 1e6 * price[0] * 1.25
    )
    return billed_input + usage["output_tokens"] / 1e6 * price[1]


def _stringify(result: Any) -> str:
    text = result if isinstance(result, str) else json.dumps(result, indent=2, default=str)
    return text[:MAX_TOOL_RESULT_CHARS]


def run_tool_loop(
    *,
    system: str,
    prompt: str,
    output_model: type[T],
    tools: list[Tool] | None = None,
    client: Any = None,
    model: str | None = None,
    max_turns: int = 8,
    max_tokens: int = 4096,
) -> tuple[T, dict]:
    """Run Claude with tools until it calls `submit_result`. Returns (parsed_output, usage)."""
    client = client or anthropic.Anthropic()
    model = model or DEFAULT_MODEL
    tools = tools or []
    by_name = {t.name: t for t in tools}
    tool_specs = [t.spec() for t in tools] + [{
        "name": SUBMIT_TOOL,
        "description": "Submit your final answer. Call this exactly once, when you are done.",
        "input_schema": output_model.model_json_schema(),
    }]

    usage = empty_usage()
    messages: list[dict] = [{"role": "user", "content": prompt}]
    # Wrap the system prompt in a cacheable block so Anthropic can reuse it across
    # turns of the same tool-loop. This cuts input costs substantially on long prompts.
    system_payload: list = [{"type": "text", "text": system,
                               "cache_control": {"type": "ephemeral"}}]

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_payload,
            messages=messages,
            tools=tool_specs,
            tool_choice={"type": "any"},
        )
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens
        usage["cache_read_tokens"] += getattr(response.usage, "cache_read_input_tokens", 0) or 0
        usage["cache_creation_tokens"] += getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        usage["llm_calls"] += 1
        messages.append({"role": "assistant", "content": response.content})

        results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            if block.name == SUBMIT_TOOL:
                try:
                    return output_model.model_validate(block.input), usage
                except ValidationError as exc:
                    results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                    "content": f"Invalid result, fix and resubmit:\n{exc}"})
                continue
            tool = by_name.get(block.name)
            if tool is None:
                results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                "content": f"Unknown tool {block.name}"})
                continue
            try:
                content, is_error = _stringify(tool.fn(**block.input)), False
            except Exception as exc:  # tool errors go back to the model, which can recover
                content, is_error = f"{type(exc).__name__}: {exc}", True
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": content, "is_error": is_error})

        if not results:
            if response.stop_reason == "max_tokens":
                raise AgentLoopError("Response hit max_tokens before calling a tool")
            if response.stop_reason == "end_turn":
                raise AgentLoopError("Model returned end_turn without calling any tool")
            messages.append({"role": "user", "content": f"Call {SUBMIT_TOOL} with your answer."})
        else:
            messages.append({"role": "user", "content": results})

    raise AgentLoopError(f"No valid {SUBMIT_TOOL} call after {max_turns} turns")
