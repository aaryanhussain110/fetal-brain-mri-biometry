from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ParameterMethod = Literal["analytic", "lookup"]


@dataclass(frozen=True)
class LookupPoint:
    ga_weeks: int
    centile_5: float
    centile_50: float
    centile_95: float


@dataclass(frozen=True)
class LookupTable:
    parameter_id: str
    source: str
    data: tuple[LookupPoint, ...]


@dataclass(frozen=True)
class MeasurementDefinition:
    parameter_id: str
    label: str
    unit: str
    group: str
    description: str
    method: ParameterMethod
    source: str


@dataclass(frozen=True)
class WorkspaceState:
    ga_weeks: int
    ga_days: int
    field_strength: str
    motion_artifact: str
    reference_profile: str
    measurements: dict[str, float | None]

    @property
    def ga_fractional(self) -> float:
        return self.ga_weeks + (self.ga_days / 7.0)


@dataclass(frozen=True)
class MeasurementResult:
    definition: MeasurementDefinition
    input_value: float | None
    mean: float | None
    std_dev: float | None
    z_score: float | None
    percentile: float | None
    marker_position: float | None
    status: str
    status_label: str
    reference_note: str | None = None


@dataclass(frozen=True)
class DifferentialCard:
    title: str
    subtitle: str
    trigger_summary: str
    items: tuple[str, ...]
    emphasis: str
    citation: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class WorkspaceViewModel:
    state: WorkspaceState
    grouped_results: tuple[tuple[str, tuple[MeasurementResult, ...]], ...]
    warning_cards: tuple[DifferentialCard, ...]
    preview_text: str
    lookup_status_summary: str
