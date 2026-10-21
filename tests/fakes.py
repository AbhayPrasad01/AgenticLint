"""A scripted stand-in for anthropic.Anthropic used by the tests (no network, no API key)."""

from __future__ import annotations

import itertools
import threading
from types import SimpleNamespace

_ids = itertools.count()


def tool_use(name: str, payload: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=f"toolu_{next(_ids)}", name=name, input=payload)


class FakeClient:
    """Routes each call to a responder chosen by the system prompt.

    `responders` maps a substring of the system prompt to a list of content-block
    lists, returned one per call for that agent.
    """

    def __init__(self, responders: dict[str, list[list]]):
        self.responders = {k: list(v) for k, v in responders.items()}
        self.calls: list[dict] = []
        self._lock = threading.Lock()
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        with self._lock:
            self.calls.append({**kwargs, "messages": list(kwargs["messages"])})  # snapshot
            # system may be a plain string or a list of content blocks (prompt-caching format)
            sys_arg = kwargs["system"]
            sys_text = sys_arg if isinstance(sys_arg, str) else " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in sys_arg
            )
            for key, queue in self.responders.items():
                if key in sys_text:
                    if not queue:
                        raise AssertionError(f"No scripted response left for {key!r}")
                    content = queue.pop(0)
                    break
            else:
                raise AssertionError(f"Unexpected system prompt: {sys_text[:60]}")
        return SimpleNamespace(content=content, stop_reason="tool_use",
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20,
                                                     cache_read_input_tokens=0,
                                                     cache_creation_input_tokens=0))


class OneShotClient:
    """Returns a single scripted response with a configurable stop_reason.

    Useful for testing edge-cases in run_tool_loop (e.g. end_turn, max_tokens)
    without going through the full graph machinery.
    """

    def __init__(self, content: list, stop_reason: str = "tool_use"):
        self._content = content
        self._stop_reason = stop_reason
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        return SimpleNamespace(
            content=self._content,
            stop_reason=self._stop_reason,
            usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )
