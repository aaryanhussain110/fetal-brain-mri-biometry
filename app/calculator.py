from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
try:
    from scipy.interpolate import PchipInterpolator
except ModuleNotFoundError:
    class PchipInterpolator:
        def __init__(self, x, y) -> None:
            self.x = np.array(x, dtype=float)
            self.y = np.array(y, dtype=float)

        def __call__(self, value: float) -> float:
            return float(np.interp(value, self.x, self.y))

try:
    from scipy.stats import norm
except ModuleNotFoundError:
    class _NormFallback:
        @staticmethod
        def cdf(value: float) -> float:
            return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    norm = _NormFallback()

from app.domain import DifferentialCard, MeasurementResult, WorkspaceState, WorkspaceViewModel
from app.reference_data import (
    DEFINITION_BY_ID,
    DIAGNOSTIC_LITERATURE_REGISTRY,
    GROUP_ORDER,
    MEASUREMENT_DEFINITIONS,
)


ANALYTIC_CURVES = {
    "tdpf": {
        "mean": (-0.01307, 2.55571, -21.71),
        "std": (0.06716, 0.547),
    },
    "csa": {
        "mean": (-0.04767, 4.20404, 1.73),
        "std": (0.01814, 5.821),
    },
}


def build_workspace_view_model(
    state: WorkspaceState,
    lookup_tables: dict,
) -> WorkspaceViewModel:
    grouped: dict[str, list[MeasurementResult]] = defaultdict(list)
    results_by_id: dict[str, MeasurementResult] = {}
    loaded_lookup_count = 0

    for definition in MEASUREMENT_DEFINITIONS:
        result = evaluate_parameter(
            parameter_id=definition.parameter_id,
            value=state.measurements.get(definition.parameter_id),
            ga_fractional=state.ga_fractional,
            lookup_tables=lookup_tables,
        )
        grouped[definition.group].append(result)
        results_by_id[definition.parameter_id] = result

        if definition.method == "lookup":
            table = lookup_tables.get(definition.parameter_id)
            if table and len(table.data) >= 2:
                loaded_lookup_count += 1

    grouped_results = tuple(
        (group_name, tuple(grouped[group_name]))
        for group_name in GROUP_ORDER
    )

    warning_cards = generate_differential_cards(state, results_by_id)
    preview_text = build_preview_text(state, grouped_results, warning_cards)

    available_lookup_total = sum(
        1 for definition in MEASUREMENT_DEFINITIONS if definition.method == "lookup"
    )
    lookup_status_summary = (
        f"{loaded_lookup_count}/{available_lookup_total} lookup-table metrics have "
        "validated centile curves loaded in this build."
    )

    return WorkspaceViewModel(
        state=state,
        grouped_results=grouped_results,
        warning_cards=warning_cards,
        preview_text=preview_text,
        lookup_status_summary=lookup_status_summary,
    )


def evaluate_parameter(
    parameter_id: str,
    value: float | None,
    ga_fractional: float,
    lookup_tables: dict,
) -> MeasurementResult:
    definition = DEFINITION_BY_ID[parameter_id]

    if value is None:
        return MeasurementResult(
            definition=definition,
            input_value=None,
            mean=None,
            std_dev=None,
            z_score=None,
            percentile=None,
            marker_position=None,
            status="empty",
            status_label="Awaiting input",
        )

    if definition.method == "analytic":
        mean, std_dev = analytic_distribution(parameter_id, ga_fractional)
        reference_note = definition.source
    else:
        table = lookup_tables.get(parameter_id)
        if not table or len(table.data) < 2:
            return MeasurementResult(
                definition=definition,
                input_value=value,
                mean=None,
                std_dev=None,
                z_score=None,
                percentile=None,
                marker_position=None,
                status="unavailable",
                status_label="Reference pending",
                reference_note="Validated lookup centile table not yet loaded.",
            )
        mean, std_dev = interpolate_distribution(table, ga_fractional)
        reference_note = table.source
        if mean is None or std_dev is None:
            return MeasurementResult(
                definition=definition,
                input_value=value,
                mean=None,
                std_dev=None,
                z_score=None,
                percentile=None,
                marker_position=None,
                status="out_of_range",
                status_label="GA outside table",
                reference_note="Gestational age lies outside the imported lookup-table range.",
            )

    z_score = (value - mean) / std_dev
    percentile = norm.cdf(z_score) * 100.0
    marker_position = float(np.clip(((z_score + 3.0) / 6.0) * 100.0, 0.0, 100.0))
    status, status_label = classify_z_score(z_score)

    return MeasurementResult(
        definition=definition,
        input_value=value,
        mean=mean,
        std_dev=std_dev,
        z_score=z_score,
        percentile=percentile,
        marker_position=marker_position,
        status=status,
        status_label=status_label,
        reference_note=reference_note,
    )


def evaluate_measurement(
    parameter_id: str,
    value: float | None,
    ga_fractional: float,
    lookup_tables: dict,
) -> MeasurementResult:
    return evaluate_parameter(
        parameter_id=parameter_id,
        value=value,
        ga_fractional=ga_fractional,
        lookup_tables=lookup_tables,
    )


def analytic_distribution(parameter_id: str, ga_fractional: float) -> tuple[float, float]:
    """
    Computes precise analytical mean and standard deviation limits 
    matching Woitek et al. (2014) polynomial regression profiles.
    """
    curve = ANALYTIC_CURVES.get(parameter_id)
    if not curve:
        return 0.0, 1.0
        
    mean_coeffs = curve["mean"]
    std_coeffs = curve["std"]
    
    mean = (mean_coeffs[0] * (ga_fractional ** 2)) + (mean_coeffs[1] * ga_fractional) + mean_coeffs[2]
    std_dev = (std_coeffs[0] * ga_fractional) + std_coeffs[1]
    
    return mean, std_dev





def interpolate_distribution(table, ga_fractional: float) -> tuple[float | None, float | None]:
    weeks = np.array([point.ga_weeks for point in table.data], dtype=float)
    centile_5 = np.array([point.centile_5 for point in table.data], dtype=float)
    centile_50 = np.array([point.centile_50 for point in table.data], dtype=float)
    centile_95 = np.array([point.centile_95 for point in table.data], dtype=float)

    lower_bound = float(np.min(weeks))
    upper_bound = float(np.max(weeks))
    if ga_fractional < lower_bound or ga_fractional > upper_bound:
        return None, None

    mu = float(PchipInterpolator(weeks, centile_50)(ga_fractional))
    lower = float(PchipInterpolator(weeks, centile_5)(ga_fractional))
    upper = float(PchipInterpolator(weeks, centile_95)(ga_fractional))
    std_dev = (((mu - lower) + (upper - mu)) / 2.0) / 1.645

    if std_dev <= 0:
        return None, None

    return mu, float(std_dev)


def classify_z_score(z_score: float) -> tuple[str, str]:
    if z_score <= -3.0:
        return "critical_low", "Critically low"
    if z_score < -2.0:
        return "low", "Below expected"
    if z_score >= 3.0:
        return "critical_high", "Critically high"
    if z_score > 2.0:
        return "high", "Above expected"
    return "normal", "Within expected"


def generate_differential_cards(
    state: WorkspaceState,
    results_by_id: dict[str, MeasurementResult],
) -> tuple[DifferentialCard, ...]:
    cards: list[DifferentialCard] = []

    left_vent = state.measurements.get("left_ventricular_diameter")
    right_vent = state.measurements.get("right_ventricular_diameter")
    severe_vm = (left_vent is not None and left_vent >= 15.0) or (
        right_vent is not None and right_vent >= 15.0
    )
    mild_moderate_vm = (
        not severe_vm
        and (
            (left_vent is not None and 10.0 <= left_vent < 15.0)
            or (right_vent is not None and 10.0 <= right_vent < 15.0)
        )
    )
    asymmetric_ventricles = (
        left_vent is not None
        and right_vent is not None
        and abs(left_vent - right_vent) > 2.0
    )

    if mild_moderate_vm:
        registry = DIAGNOSTIC_LITERATURE_REGISTRY["mild_moderate_ventriculomegaly"]
        cards.append(
            DifferentialCard(
                title=registry["condition"],
                subtitle="Atrial diameter between 10 mm and 15 mm",
                trigger_summary=registry["trigger"],
                items=(
                    "Isolated or idiopathic (~8%): neurodevelopmental delay risk is relatively low in isolated mild VM.",
                    "Associated CNS or extracranial anomaly: a key structural workup target when dilation is not clearly isolated.",
                    "Chromosomal abnormality (~5-15%): aneuploidy remains an important underlying cause.",
                ),
                emphasis="warning",
                citation=registry["citation"],
                url=registry["url"],
            )
        )

    if severe_vm:
        registry = DIAGNOSTIC_LITERATURE_REGISTRY["severe_ventriculomegaly"]
        cards.append(
            DifferentialCard(
                title="Severe Ventriculomegaly",
                subtitle="Atrial diameter threshold crossed",
                trigger_summary="Left or right atrial diameter is at least 15 mm.",
                items=(
                    "Aqueductal stenosis (~20%): common obstructive hydrocephalus source.",
                    "Associated CNS or non-CNS anomaly (high): severe dilation is frequently not isolated.",
                    "Chromosomal abnormality (significant): includes trisomy 13, 18, and 21.",
                    "Congenital infection (e.g., CMV) (~1-5%): important infectious exclusion.",
                    "Isolated or idiopathic (~10-20%): diagnosis of exclusion after full workup.",
                ),
                emphasis="warning",
                citation=registry["citation"],
                url=registry["url"],
            )
        )

    if asymmetric_ventricles:
        registry = DIAGNOSTIC_LITERATURE_REGISTRY["asymmetric_ventricles"]
        cards.append(
            DifferentialCard(
                title=registry["condition"],
                subtitle="Right-left ventricular difference exceeds 2 mm",
                trigger_summary=registry["trigger"],
                items=(
                    "Often benign if isolated, but warrants follow-up because progression toward true ventriculomegaly can occur.",
                    "Progression risk is reported around 37-46% in the cited MRI cohort.",
                ),
                emphasis="warning",
                citation=registry["citation"],
                url=registry["url"],
            )
        )

    tdpf = results_by_id.get("tdpf")
    csa = results_by_id.get("csa")
    joint_probability = chiari_joint_probability(tdpf, csa)
    if (
        tdpf
        and csa
        and tdpf.z_score is not None
        and csa.z_score is not None
        and tdpf.z_score < -2.0
        and csa.z_score < -2.0
        and joint_probability is not None
        and joint_probability > 0.5
    ):
        registry = DIAGNOSTIC_LITERATURE_REGISTRY["chiari_ii_ontd"]
        cards.append(
            DifferentialCard(
                title="Chiari II Malformation / Open NTD",
                subtitle="Posterior fossa pattern match",
                trigger_summary=(
                    f"TDPF Z {tdpf.z_score:+.2f}, CSA Z {csa.z_score:+.2f}, "
                    f"joint ONTD probability {joint_probability:.0%}."
                ),
                items=(
                    "Chiari II malformation / open neural tube defect: about 85-90% likelihood.",
                    "Closed neural tube defect: about 5-10% likelihood; typically milder posterior fossa distortion.",
                    "Severe vermian hypoplasia / Dandy-Walker continuum: about 3-5% likelihood; CSA is usually not comparably narrowed.",
                    "Benign small posterior fossa: under 1%; residual diagnosis once the full pattern is checked.",
                ),
                emphasis="critical",
                citation=registry["citation"],
                url=registry["url"],
            )
        )

    return tuple(cards)


def chiari_joint_probability(
    tdpf: MeasurementResult | None,
    csa: MeasurementResult | None,
) -> float | None:
    posteriors = chiari_group_posteriors(tdpf, csa)
    if posteriors is None:
        return None
    return posteriors["ontd"]


def chiari_group_posteriors(
    tdpf: MeasurementResult | None,
    csa: MeasurementResult | None,
) -> dict[str, float] | None:
    if not tdpf or not csa or tdpf.z_score is None or csa.z_score is None:
        return None

    observation = np.array([tdpf.z_score, csa.z_score], dtype=float)
    groups = {
        "controls": (
            np.array([0.0, 0.0], dtype=float),
            covariance_matrix(1.0, 1.0, 0.0),
        ),
        "ontd": (
            np.array([-3.6, -2.6], dtype=float),
            covariance_matrix(0.9, 1.1, 0.54),
        ),
        "cntd": (
            np.array([-1.4, -0.6], dtype=float),
            covariance_matrix(1.0, 1.0, 0.0),
        ),
    }

    weights: dict[str, float] = {}
    for group_name, (mean, covariance) in groups.items():
        distance_squared = mahalanobis_squared(observation, mean, covariance)
        weights[group_name] = math.exp(-0.5 * distance_squared)

    total = sum(weights.values())
    if total <= 0:
        return None

    return {
        group_name: weight / total
        for group_name, weight in weights.items()
    }


def covariance_matrix(std_x: float, std_y: float, correlation: float) -> np.ndarray:
    covariance = correlation * std_x * std_y
    return np.array(
        [[std_x**2, covariance], [covariance, std_y**2]],
        dtype=float,
    )


def multivariate_normal_pdf(
    observation: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> float:
    dimension = mean.shape[0]
    centered = observation - mean
    inverse = np.linalg.inv(covariance)
    mahalanobis_squared = float(centered.T @ inverse @ centered)
    determinant = float(np.linalg.det(covariance))
    normalizer = math.sqrt(((2 * math.pi) ** dimension) * determinant)
    return math.exp(-0.5 * mahalanobis_squared) / normalizer


def mahalanobis_squared(
    observation: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> float:
    centered = observation - mean
    inverse = np.linalg.inv(covariance)
    return float(centered.T @ inverse @ centered)


def build_preview_text(
    state: WorkspaceState,
    grouped_results: tuple[tuple[str, tuple[MeasurementResult, ...]], ...],
    warning_cards: tuple[DifferentialCard, ...],
) -> str:
    lines = [
        "Fetal Brain MRI Biometry",
        f"GA: {state.ga_weeks}w {state.ga_days}d ({state.ga_fractional:.2f} weeks)",
        (
            f"Quality context: {state.field_strength} | Motion: {state.motion_artifact} | "
            f"Reference profile: {state.reference_profile}"
        ),
        "",
    ]

    for group_name, results in grouped_results:
        populated = [result for result in results if result.input_value is not None]
        if not populated:
            continue

        lines.append(group_name)
        for result in populated:
            value = f"{result.input_value:.1f} {result.definition.unit}"
            if result.z_score is not None and result.percentile is not None:
                lines.append(
                    f"{result.definition.label}: {value} "
                    f"(Z: {result.z_score:+.2f}, percentile: {result.percentile:.0f})"
                )
            else:
                lines.append(
                    f"{result.definition.label}: {value} "
                    f"({result.reference_note or 'reference unavailable'})"
                )
        lines.append("")

    if warning_cards:
        lines.append("Clinical Warnings")
        for card in warning_cards:
            lines.append(f"{card.title}: {card.trigger_summary}")
        lines.append("")

    return "\n".join(lines).strip()
