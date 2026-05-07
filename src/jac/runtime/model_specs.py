"""Per-model spec lookup (max context window). Data lives in `jac.data/model_specs.toml`."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


@dataclass(frozen=True, slots=True)
class ModelSpec:
    max_context: int


@lru_cache(maxsize=1)
def _load() -> tuple[dict[str, ModelSpec], ModelSpec]:
    raw = tomllib.loads(
        files("jac.data").joinpath("model_specs.toml").read_text(encoding="utf-8")
    )
    default = ModelSpec(max_context=int(raw["defaults"]["max_context"]))
    models = {
        key: ModelSpec(max_context=int(entry["max_context"]))
        for key, entry in (raw.get("models") or {}).items()
    }
    return models, default


def spec_for(model_ref: str) -> ModelSpec:
    """Look up the spec for a model_ref. Falls back to defaults if unknown."""
    models, default = _load()
    return models.get(model_ref, default)
