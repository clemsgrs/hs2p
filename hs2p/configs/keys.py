"""Reject configuration keys that ``default.yaml`` does not define.

``default.yaml`` is the only schema: every key path present in a config must exist there.
Missing keys are never an error (defaults apply, and partial programmatic adapters stay
valid). The user's label vocabulary under ``tiling.masks`` is open, and ``wandb`` is not
validated.
"""

import dataclasses
import difflib
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .loader import default_config

#: Maps whose keys are user label names rather than schema keys.
OPEN_MAPS = frozenset(
    {
        "tiling.masks.pixel_mapping",
        "tiling.masks.colors",
        "tiling.masks.min_coverage",
    }
)
#: Sections whose contents are not validated.
UNCHECKED_SECTIONS = frozenset({"wandb"})
#: Retired keys, with the migration message that replaces the generic unknown-key error.
RETIRED_KEYS = {
    "tiling.sampling_params": (
        "tiling.sampling_params is no longer supported; move it to tiling.masks: "
        "pixel_mapping stays pixel_mapping, color_mapping becomes colors, and "
        "tissue_percentage becomes min_coverage."
    ),
}

_SCHEMA: dict[str, Any] = OmegaConf.to_container(default_config, resolve=False)


def reject_unknown_config_keys(cfg: Any, *, path: str = "") -> None:
    """Raise one ``ValueError`` naming every key under ``cfg`` that the schema lacks.

    ``path`` is the dotted location of ``cfg`` in the full config (``""`` for the root).
    Interpolations are never evaluated.
    """
    unknown = _unknown_keys(cfg, path=path, schema=_schema_at(path))
    for unknown_path, _ in unknown:
        if unknown_path in RETIRED_KEYS:
            raise ValueError(RETIRED_KEYS[unknown_path])
    if unknown:
        raise ValueError("\n".join(message for _, message in unknown))


def _schema_at(path: str) -> Any:
    schema: Any = _SCHEMA
    for part in path.split(".") if path else ():
        schema = schema[part]
    return schema


def _unknown_keys(node: Any, *, path: str, schema: Any) -> list[tuple[str, str]]:
    """Collect ``(path, message)`` for every unknown key under ``node``."""
    if not isinstance(schema, dict) or path in OPEN_MAPS or path in UNCHECKED_SECTIONS:
        return []
    members = _members(node)
    if members is None:
        return []
    unknown: list[tuple[str, str]] = []
    for key, value in members.items():
        key = str(key)
        child_path = f"{path}.{key}" if path else key
        if key not in schema:
            unknown.append(
                (child_path, _describe_unknown(key, path=path, schema=schema))
            )
            continue
        unknown.extend(_unknown_keys(value, path=child_path, schema=schema[key]))
    return unknown


def _members(node: Any) -> dict[str, Any] | None:
    """Return a config section's members without resolving interpolations.

    Sections may be ``DictConfig`` (file/CLI loading), plain mappings, or the
    ``SimpleNamespace`` and dataclass adapters programmatic callers build. Anything else
    has no enumerable members and is not checked.
    """
    if isinstance(node, DictConfig):
        return OmegaConf.to_container(node, resolve=False)
    if isinstance(node, Mapping):
        return dict(node)
    if isinstance(node, SimpleNamespace):
        return vars(node)
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        return {field.name: getattr(node, field.name) for field in dataclasses.fields(node)}
    return None


def _describe_unknown(key: str, *, path: str, schema: dict[str, Any]) -> str:
    prefix = f"{path}." if path else ""
    message = f"Unknown config key {prefix}{key}"
    suggestion = _closest(key, list(schema))
    if suggestion is not None:
        return f"{message} (did you mean {prefix}{suggestion}?)"
    level = path or "the top level"
    return f"{message} (valid keys at {level}: {', '.join(sorted(schema))})"


def _closest(key: str, candidates: list[str]) -> str | None:
    """Closest candidate by spelling, also matching ``key`` against each word of a
    candidate so ``spacng`` finds ``requested_spacing_um``."""

    def score(candidate: str) -> float:
        words = [candidate, *candidate.split("_")]
        return max(difflib.SequenceMatcher(None, key, word).ratio() for word in words)

    best = max(candidates, key=score, default=None)
    if best is None or score(best) < 0.75:
        return None
    return best
