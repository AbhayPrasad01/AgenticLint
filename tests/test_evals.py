import json

from evals.cases import CASES
from evals.run_eval import percentile, run_case, score_case, summarize, write_results
from review_agent.diff import parse_unified_diff
from tests.fakes import FakeClient, tool_use


def test_thirty_unique_cases_with_bug_on_an_added_line():
    assert len(CASES) == 30
    assert len({c.id for c in CASES}) == 30
    for c in CASES:
        assert c.bug_line in parse_unified_diff(c.diff)[c.path].added_lines, c.id


def test_score_case_counts_hits_within_tolerance():
    c = CASES[0]
    comments = [
        {"path": c.path, "line": c.bug_line + 1},    # hit
        {"path": c.path, "line": c.bug_line + 10},   # false positive
        {"path": "other.py", "line": c.bug_line},    # false positive
    ]
    s = score_case(c, comments)
    assert s == {"detected": True, "true_positive_comments": 1,
                 "false_positive_comments": 2, "total_comments": 3}


def test_run_case_and_summary_end_to_end_with_fake_model():
    c = CASES[0]
    finding = {"path": c.path, "line": c.bug_line, "severity": "critical", "category": "security",
               "title": "SQL injection", "explanation": "x"}
    client = FakeClient({
        "planning agent": [[tool_use("submit_result", {"files": [], "focus_areas": []})]],
        "senior engineer": [[tool_use("submit_result", {"findings": [finding]})]],
        "lead reviewer": [[tool_use("submit_result", {"summary": "s", "comments": [finding]})]],
    })
    result = run_case(c, client, "claude-haiku-4-5-20251001")
    assert result["detected"] and result["error"] is None
    summary = summarize([result], "claude-haiku-4-5-20251001")
    assert summary["detection_rate"] == 1.0
    assert summary["false_positive_rate"] == 0
    assert summary["avg_cost_usd_per_review"] is not None
    # Usage must always have the full 5-key schema
    for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens", "llm_calls"):
        assert key in result["usage"], f"Missing usage key: {key}"


def test_run_case_error_path_returns_full_usage_schema(tmp_path):
    """When the graph raises, run_case must still return a 5-key usage dict."""
    c = CASES[0]
    # Empty responders -> FakeClient raises AssertionError (graph fails)
    client = FakeClient({
        "planning agent": [[tool_use("submit_result", {"files": [], "focus_areas": []})]],
        "senior engineer": [],  # exhausted queue -> AssertionError -> AgentLoopError path
    })
    result = run_case(c, client, "claude-haiku-4-5-20251001")
    assert result["error"] is not None        # confirmed failure
    assert result["detected"] is False        # miss, not a crash
    usage = result["usage"]
    for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens", "llm_calls"):
        assert key in usage, f"Missing usage key on error path: {key}"



def test_percentile():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([], 0.95) == 0.0


def test_write_results_to_custom_dir(tmp_path):
    """write_results() must create the output dir and write .json + .md files."""
    summary = {"model": "test-model", "cases": 1, "detection_rate": 1.0,
               "false_positive_rate": 0.0, "comments_per_review": 1.0,
               "avg_tokens_per_review": 100, "avg_cost_usd_per_review": 0.001,
               "latency_p50_s": 1.0, "latency_p95_s": 1.0, "errors": 0,
               "detection_by_category": {}}
    results = [{"id": "c01", "category": "security", "detected": True,
                "false_positive_comments": 0, "latency_s": 1.0}]
    out_dir = tmp_path / "custom" / "nested"
    base = write_results(summary, results, out_dir)

    assert out_dir.is_dir(), "output directory must be created"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    assert json_path.is_file() and md_path.is_file()

    data = json.loads(json_path.read_text())
    assert data["summary"]["model"] == "test-model"
    assert data["results"][0]["id"] == "c01"

    md = md_path.read_text()
    assert "test-model" in md and "c01" in md


def test_write_results_sanitizes_model_name(tmp_path):
    """A model name with '/' must not create nested directories in the output path."""
    summary = {"model": "org/my-model:latest", "cases": 0, "detection_rate": 0.0,
               "false_positive_rate": 0.0, "comments_per_review": 0.0,
               "avg_tokens_per_review": 0, "avg_cost_usd_per_review": None,
               "latency_p50_s": 0.0, "latency_p95_s": 0.0, "errors": 0,
               "detection_by_category": {}}
    base = write_results(summary, [], tmp_path)

    # The base path must live directly inside tmp_path (no unexpected subdirs)
    assert base.parent == tmp_path, f"Expected parent {tmp_path}, got {base.parent}"
    # The filename must not contain '/' or ':'
    assert "/" not in base.name and ":" not in base.name
    # Both output files must exist as regular files
    assert base.with_suffix(".json").is_file()
    assert base.with_suffix(".md").is_file()
    # The JSON still stores the original model name
    data = json.loads(base.with_suffix(".json").read_text())
    assert data["summary"]["model"] == "org/my-model:latest"
