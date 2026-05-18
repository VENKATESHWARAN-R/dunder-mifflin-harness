"""Slice 3 smoke: exercise the full tool wrap chain end-to-end.

Builds a FunctionModel-backed Scott agent under YOLO approval (so no
prompting), wires `ScottDeps(run_id, tasks_repo)` against a fresh SQLite
state.db, and drives a deterministic turn sequence:

    add_task("write the report") → list_tasks → complete_task → list_tasks

After the run finishes, the SQLite `tasks` table is queried directly to
confirm each turn's effect actually committed. This is the lab analog of
`tests/agents/test_factory.py::test_build_agent_runs_task_tool_via_function_model`
— same wrap chain, end-to-end, against a real on-disk DB.

Usage:
    uv run python lab/scripts/tools_hello.py [path]
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from jac.agents import build_agent
from jac.runtime.approvals import ApprovalMode, ApprovalPolicy
from jac.state import open_state_store
from jac.tools.types import ScottDeps


async def main(db_path: Path) -> int:
    store = await open_state_store(db_path)
    try:
        run_id = uuid4().hex
        await store.runs.create(run_id, prompt="tools_hello smoke")
        print(f"run_id = {run_id}")

        first_task_id: dict[str, str] = {}

        async def behaviour(messages, info):  # noqa: ANN001 — FunctionModel signature
            # Step through the deterministic script. Each model "turn" picks the
            # next tool call based on how many messages have come back so far.
            tool_results = [
                part
                for msg in messages
                for part in getattr(msg, "parts", [])
                if type(part).__name__ == "ToolReturnPart"
            ]
            step = len(tool_results)
            if step == 0:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="add_task",
                            args={"title": "write the report"},
                        )
                    ]
                )
            if step == 1:
                # Capture the task_id from the previous return for completion.
                payload = tool_results[-1].content
                first_task_id["id"] = payload.task.task_id  # type: ignore[union-attr]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="list_tasks", args={})]
                )
            if step == 2:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="complete_task",
                            args={"task_id": first_task_id["id"]},
                        )
                    ]
                )
            if step == 3:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="list_tasks", args={})]
                )
            return ModelResponse(parts=[TextPart(content="done")])

        agent = build_agent(
            model=FunctionModel(behaviour),
            approval_policy=ApprovalPolicy(mode=ApprovalMode.YOLO),
        )
        deps = ScottDeps(run_id=run_id, tasks_repo=store.tasks)

        result = await agent.run("go", deps=deps)
        print(f"agent output  : {result.output!r}")

        rows = await store.tasks.list_for_run(run_id)
        print(f"task rows     : {len(rows)}")
        for row in rows:
            print(f"  - {row.task_id[:8]}  {row.status:11s}  {row.title}")

        assert len(rows) == 1, "exactly one task should have been created"
        assert rows[0].status == "completed", "task should be marked completed"
        print("OK")
        return 0
    finally:
        await store.close()


def _cli() -> int:
    if len(sys.argv) > 1:
        return asyncio.run(main(Path(sys.argv[1])))
    with tempfile.TemporaryDirectory() as tmp:
        return asyncio.run(main(Path(tmp) / "state.db"))


if __name__ == "__main__":
    sys.exit(_cli())
