"""Shared data models for the agent pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PipelineResult:
    image_path: str
    ground_truth_class: str | None
    math_stats: dict[str, Any]
    vlm_result: dict[str, Any] | None
    final_decision: str | None
    decision_source: str | None
    agreement: bool | None
    processed_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
