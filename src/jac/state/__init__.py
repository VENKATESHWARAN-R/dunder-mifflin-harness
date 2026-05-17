"""SQLite-backed persistence layer for JAC.

See `docs/contracts/STATE_SCHEMA.md` for the schema (v1.5) and
`docs/contracts/SUBSTRATE.md` for what belongs here vs. on disk.
"""

from jac.state.agent_configs import (
    AgentConfigRow,
    AgentConfigsRepo,
    fetch_per_run_override,
)
from jac.state.attempts import AttemptRow, AttemptsRepo, RunTotals
from jac.state.db import StateStore, open_state_store
from jac.state.mcp_servers import McpServerRow, McpServersRepo
from jac.state.messages import MessageRow, MessagesRepo
from jac.state.run_mcp_servers import (
    ActiveMcpServerRow,
    RunMcpServerRow,
    RunMcpServersRepo,
)
from jac.state.run_skills import ActiveSkillRow, RunSkillRow, RunSkillsRepo
from jac.state.runs import RunRow, RunsRepo
from jac.state.seeder import SeedResult, seed_workspace
from jac.state.skills import SkillRow, SkillsRepo
from jac.state.tasks import TaskRow, TasksRepo

__all__ = [
    "ActiveMcpServerRow",
    "ActiveSkillRow",
    "AgentConfigRow",
    "AgentConfigsRepo",
    "AttemptRow",
    "AttemptsRepo",
    "McpServerRow",
    "McpServersRepo",
    "MessageRow",
    "MessagesRepo",
    "RunMcpServerRow",
    "RunMcpServersRepo",
    "RunRow",
    "RunsRepo",
    "RunSkillRow",
    "RunSkillsRepo",
    "RunTotals",
    "SeedResult",
    "SkillRow",
    "SkillsRepo",
    "StateStore",
    "TaskRow",
    "TasksRepo",
    "fetch_per_run_override",
    "open_state_store",
    "seed_workspace",
]
