"""Slice 2 smoke: open a fresh state.db, write some rows, read them back.

Usage:
    uv run python lab/scripts/state_hello.py [path]

If no path is given, writes to a temp directory and cleans up. The script
exercises every M1 active table at least once and prints summary counts.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from jac.state import fetch_per_run_override, open_state_store


async def main(db_path: Path) -> int:
    store = await open_state_store(db_path)
    try:
        run_id = uuid4().hex
        await store.runs.create(run_id, prompt="state_hello smoke")

        await store.agent_configs.create(
            run_id=run_id,
            role="manager",
            model_tier="worker",
            model_override="anthropic:claude-sonnet-4-6",
        )

        override = await fetch_per_run_override(store, run_id, role="manager")
        print(
            f"[state_hello] per-run override: tier={override.tier} "
            f"model={override.model_override}"
        )

        attempt_id = uuid4().hex
        await store.attempts.create(
            attempt_id=attempt_id,
            run_id=run_id,
            model="anthropic:claude-sonnet-4-6",
            tier="worker",
        )
        await store.attempts.update_usage(
            attempt_id,
            tokens_in=120,
            tokens_out=80,
            requests=1,
            tool_calls=0,
            cost=0.005,
            duration_ms=420,
        )

        for title in ("scaffold project", "wire tests", "ship"):
            await store.tasks.create(run_id=run_id, title=title)

        await store.messages.append(run_id, role="user", content="hello")
        await store.messages.append(run_id, role="assistant", content="hi back")

        totals = await store.attempts.totals_for_run(run_id)
        tasks = await store.tasks.list_for_run(run_id)
        msg_count = await store.messages.count_for_run(run_id)

        print(f"[state_hello] run_id={run_id}")
        print(f"[state_hello] tasks={[t.title for t in tasks]}")
        print(
            f"[state_hello] messages={msg_count} attempts={totals.attempts} "
            f"tokens_in={totals.tokens_in} tokens_out={totals.tokens_out} "
            f"cost={totals.cost:.4f}"
        )
        return 0
    finally:
        await store.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = Path(sys.argv[1]).expanduser().resolve()
        raise SystemExit(asyncio.run(main(target)))
    with tempfile.TemporaryDirectory(prefix="jac-state-hello-") as tmp:
        raise SystemExit(asyncio.run(main(Path(tmp) / "state.db")))
