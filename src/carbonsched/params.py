"""Cited, sweepable parameters.

Every number in ``configs/*.yaml`` may be written either as a bare number or as a
mapping::

    {value: 0.09, low: 0.03, high: 0.15, source: "Llama 3 paper, Table 5", status: verified}

``status`` is one of ``verified`` (number read from the cited source),
``derived`` (computed from verified numbers; the derivation is in ``source``),
or ``assumption`` (engineering judgement; always swept in sensitivity studies).

`resolve` turns such a tree into plain numbers and records every `Param`, so
experiments can sweep ranges and `paper/parameters.md` can be generated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

STATUSES = ("verified", "derived", "assumption")


@dataclass(frozen=True)
class Param:
    path: str
    value: float
    low: float | None = None
    high: float | None = None
    source: str = ""
    status: str = "assumption"
    unit: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"{self.path}: bad status {self.status!r}")
        if self.low is not None and self.high is not None and not (self.low <= self.value <= self.high):
            raise ValueError(f"{self.path}: value {self.value} outside [{self.low}, {self.high}]")


def _is_param(node: Any) -> bool:
    return isinstance(node, dict) and "value" in node and isinstance(node["value"], (int, float))


def resolve(tree: Any, registry: dict[str, Param] | None = None, prefix: str = "",
            overrides: dict[str, float] | None = None) -> Any:
    """Return ``tree`` with every Param mapping replaced by its value.

    ``overrides`` maps dotted paths to replacement values (used for sweeps).
    """
    overrides = overrides or {}
    if _is_param(tree):
        p = Param(path=prefix, value=float(tree["value"]), low=tree.get("low"), high=tree.get("high"),
                  source=tree.get("source", ""), status=tree.get("status", "assumption"),
                  unit=tree.get("unit", ""))
        if registry is not None:
            registry[prefix] = p
        return float(overrides.get(prefix, p.value))
    if isinstance(tree, dict):
        return {k: resolve(v, registry, f"{prefix}.{k}" if prefix else str(k), overrides) for k, v in tree.items()}
    if isinstance(tree, list):
        return [resolve(v, registry, f"{prefix}[{i}]", overrides) for i, v in enumerate(tree)]
    if isinstance(tree, (int, float)) and not isinstance(tree, bool) and prefix in overrides:
        return float(overrides[prefix])
    return tree


def load_yaml(path: str | Path, registry: dict[str, Param] | None = None,
              overrides: dict[str, float] | None = None) -> Any:
    raw = yaml.safe_load(Path(path).read_text())
    stem = Path(path).stem
    return resolve(raw, registry, stem, {k: v for k, v in (overrides or {}).items()})


def parameters_table(registry: dict[str, Param]) -> str:
    """Markdown table of all cited parameters (for paper/parameters.md)."""
    lines = ["| Parameter | Value | Range | Unit | Status | Source |", "|---|---|---|---|---|---|"]
    for p in sorted(registry.values(), key=lambda p: p.path):
        rng = f"[{p.low:g}, {p.high:g}]" if p.low is not None and p.high is not None else ""
        lines.append(f"| `{p.path}` | {p.value:g} | {rng} | {p.unit} | {p.status} | {p.source} |")
    return "\n".join(lines) + "\n"
