"""C7 /context growth rendering."""

from __future__ import annotations

from dataclasses import replace

from rich.console import Console

from jac.cli.renderer import Renderer
from jac.runtime.model_specs import spec_for
from jac.state.attempts import AttemptRow


def test_render_context_growth_empty_attempts() -> None:
    r = Renderer(console=Console(force_terminal=True, width=80, height=24, record=True))
    r.render_context_growth(
        run_id="abc",
        message_count=0,
        cwd="/tmp",
        attached="",
        last_ctx=0,
        model_max=spec_for("anthropic:claude-sonnet-4-6").max_context,
        attempt_rows=[],
    )
    out = r.console.export_text()
    assert "run_id: abc" in out


def test_render_context_growth_deltas_and_headroom() -> None:
    r = Renderer(
        console=Console(force_terminal=True, width=100, height=40, record=True)
    )
    mx = spec_for("anthropic:claude-sonnet-4-6").max_context
    base = AttemptRow(
        attempt_id="a",
        task_id=None,
        run_id="r",
        parent_attempt_id=None,
        call_type="agent",
        role="manager",
        model="m",
        tier="worker",
        tokens_in=0,
        tokens_out=0,
        requests=0,
        tool_calls=0,
        cost=0.0,
        duration_ms=0,
        eval_score=None,
        eval_passed=None,
        eval_feedback=None,
        status="passed",
        created_at="t",
    )
    rows = [
        replace(base, attempt_id="1", tokens_in=1000, role="manager"),
        replace(base, attempt_id="2", tokens_in=500, role="manager"),
        replace(base, attempt_id="3", tokens_in=200, role="builder"),
    ]
    last_ctx = 1700
    r.render_context_growth(
        run_id="r",
        message_count=3,
        cwd="/w",
        attached="",
        last_ctx=last_ctx,
        model_max=mx,
        attempt_rows=rows,
    )
    text = r.console.export_text()
    assert "headroom" in text
    headroom = mx - last_ctx
    assert f"{headroom:,}" in text
