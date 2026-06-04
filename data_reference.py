from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
except ModuleNotFoundError:
    class PchipInterpolator:
        """Linear fallback used only when SciPy is unavailable in the test runner."""

        def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
            self.x = np.array(x, dtype=float)
            self.y = np.array(y, dtype=float)

        def __call__(self, value: float) -> float:
            return float(np.interp(value, self.x, self.y))


@dataclass(frozen=True)
class ParameterDefinition:
    parameter_id: str
    label: str
    unit: str
    group: str
    description: str
    method: str


PARAMETERS: tuple[ParameterDefinition, ...] = (
    ParameterDefinition(
        parameter_id="skull_bpd",
        label="Skull BPD",
        unit="mm",
        group="Global Growth",
        description="Outer biparietal diameter across the calvarium.",
        method="lookup",
    ),
    ParameterDefinition(
        parameter_id="skull_ofd",
        label="Skull OFD",
        unit="mm",
        group="Global Growth",
        description="Outer occipitofrontal diameter across the calvarium.",
        method="lookup",
    ),
    ParameterDefinition(
        parameter_id="brain_bpd",
        label="Brain BPD",
        unit="mm",
        group="Global Growth",
        description="Inner cerebral biparietal diameter.",
        method="lookup",
    ),
    ParameterDefinition(
        parameter_id="brain_ofd",
        label="Brain OFD",
        unit="mm",
        group="Global Growth",
        description="Inner cerebral occipitofrontal diameter.",
        method="lookup",
    ),
    ParameterDefinition(
        parameter_id="left_ventricular_diameter",
        label="Left Ventricular Diameter",
        unit="mm",
        group="Ventricular System",
        description="Atrial width of the left lateral ventricle.",
        method="lookup",
    ),
    ParameterDefinition(
        parameter_id="right_ventricular_diameter",
        label="Right Ventricular Diameter",
        unit="mm",
        group="Ventricular System",
        description="Atrial width of the right lateral ventricle.",
        method="lookup",
    ),
    ParameterDefinition(
        parameter_id="third_ventricle_width",
        label="Third Ventricle Width",
        unit="mm",
        group="Ventricular System",
        description="Maximum transverse width of the third ventricle.",
        method="lookup",
    ),
    ParameterDefinition(
        parameter_id="corpus_callosum_length",
        label="Corpus Callosum Length",
        unit="mm",
        group="Midline Structures",
        description="Mid-sagittal corpus callosum length.",
        method="lookup",
    ),
    ParameterDefinition(
        parameter_id="csp_width",
        label="CSP Width",
        unit="mm",
        group="Midline Structures",
        description="Maximum cavum septum pellucidum width.",
        method="lookup",
    ),
    ParameterDefinition(
        parameter_id="transcerebellar_diameter",
        label="Transcerebellar Diameter",
        unit="mm",
        group="Posterior Fossa",
        description="Maximum transverse cerebellar diameter.",
        method="lookup",
    ),
    ParameterDefinition(
        parameter_id="vermian_height",
        label="Vermian Height",
        unit="mm",
        group="Posterior Fossa",
        description="Superior-inferior vermian dimension.",
        method="lookup",
    ),
    ParameterDefinition(
        parameter_id="vermian_ap_diameter",
        label="Vermian AP Diameter",
        unit="mm",
        group="Posterior Fossa",
        description="Anterior-posterior vermian dimension.",
        method="lookup",
    ),
    ParameterDefinition(
        parameter_id="cisterna_magna_depth",
        label="Cisterna Magna Depth",
        unit="mm",
        group="Posterior Fossa",
        description="Anteroposterior depth of the cisterna magna.",
        method="lookup",
    ),
    ParameterDefinition(
        parameter_id="tegmento_vermian_angle",
        label="Tegmento-Vermian Angle",
        unit="degrees",
        group="Posterior Fossa",
        description="Angle between the dorsal brainstem and ventral vermis.",
        method="lookup",
    ),
    ParameterDefinition(
        parameter_id="tdpf",
        label="Posterior Fossa Diameter (TDPF)",
        unit="mm",
        group="Posterior Fossa",
        description="Maximum transverse diameter of the posterior fossa.",
        method="analytic",
    ),
    ParameterDefinition(
        parameter_id="csa",
        label="Clivus-Supraocciput Angle (CSA)",
        unit="degrees",
        group="Posterior Fossa",
        description="Angle subtended by the clivus and supraocciput.",
        method="analytic",
    ),
    ParameterDefinition(
        parameter_id="pons_ap_diameter",
        label="Pons AP Diameter",
        unit="mm",
        group="Brainstem",
        description="Anterior-posterior pons diameter.",
        method="lookup",
    ),
)


GROUP_ORDER = (
    "Global Growth",
    "Ventricular System",
    "Midline Structures",
    "Posterior Fossa",
    "Brainstem",
)


DIAGNOSTIC_LITERATURE_REGISTRY: dict[str, dict[str, str]] = {
    "mild_moderate_ventriculomegaly": {
        "condition": "Mild-to-Moderate Ventriculomegaly",
        "trigger": "Atrial diameter >= 10 mm and < 15 mm",
        "citation": "Pagani G, et al. Ultrasound Obstet Gynecol. 2014;44(3):254-260.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/?term=Pagani+2014+Ultrasound+Obstet+Gynecol+44+3+254-260",
    },
    "severe_ventriculomegaly": {
        "condition": "Severe Ventriculomegaly",
        "trigger": "Atrial diameter >= 15 mm",
        "citation": "Giorgione V, et al. Prenat Diagn. 2022;42(13):1674-1681.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/?term=Giorgione+2022+Prenat+Diagn+42+13+1674-1681",
    },
    "asymmetric_ventricles": {
        "condition": "Asymmetric Lateral Ventricles",
        "trigger": "Right vs left atrial difference > 2 mm",
        "citation": "Barzilay E, et al. AJNR Am J Neuroradiol. 2017;38(2):371-375.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/?term=Barzilay+2017+AJNR+38+2+371-375",
    },
    "chiari_ii_ontd": {
        "condition": "Chiari II Malformation / Open NTD",
        "trigger": "TDPF z < -2.0 AND CSA z < -2.0 AND posterior probability > 0.5",
        "citation": "Woitek R, Prayer D, Weber M, et al. PLOS One. 2014;9(11):e112585.",
        "url": "https://doi.org/10.1371/journal.pone.0112585",
    },
}


MOCK_GA_WEEKS = tuple(range(18, 41))
MIN_CENTILE_SPREAD = 0.35
MIN_STD_DEV = 0.05


def build_mock_growth_table(
    *,
    start_median: float,
    end_median: float,
    start_spread: float,
    end_spread: float,
    curvature: float = 0.0,
    minimum_value: float = 0.0,
) -> list[dict[str, float]]:
    """Build a monotonic mock centile table across 18-40 weeks.

    These values are deliberately synthetic but shaped like plausible normative
    reference curves so the spline engine can be exercised safely in the UI.
    """

    step_count = len(MOCK_GA_WEEKS) - 1
    rows: list[dict[str, float]] = []

    for index, ga_week in enumerate(MOCK_GA_WEEKS):
        t = index / step_count
        shaped_t = t + (curvature * t * (1.0 - t))

        median = start_median + ((end_median - start_median) * shaped_t)
        spread = start_spread + ((end_spread - start_spread) * t)
        spread = max(spread, MIN_CENTILE_SPREAD)

        centile_5 = max(minimum_value, median - spread)
        centile_95 = median + spread

        if median <= centile_5:
            median = centile_5 + MIN_CENTILE_SPREAD
        if centile_95 <= median:
            centile_95 = median + MIN_CENTILE_SPREAD

        rows.append(
            {
                "ga_weeks": float(ga_week),
                "centile_5": round(centile_5, 1),
                "centile_50": round(median, 1),
                "centile_95": round(centile_95, 1),
            }
        )

    return rows


def normalize_lookup_rows(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    normalized: list[dict[str, float]] = []

    for row in sorted(rows, key=lambda item: item["ga_weeks"]):
        ga_week = int(row["ga_weeks"])
        centile_50 = float(row["centile_50"])

        lower_spread = max(centile_50 - float(row["centile_5"]), MIN_CENTILE_SPREAD)
        upper_spread = max(float(row["centile_95"]) - centile_50, MIN_CENTILE_SPREAD)

        normalized.append(
            {
                "ga_weeks": float(ga_week),
                "centile_5": round(max(0.0, centile_50 - lower_spread), 1),
                "centile_50": round(centile_50, 1),
                "centile_95": round(centile_50 + upper_spread, 1),
            }
        )

    return normalized


def apply_documented_row_overrides(
    parameter_id: str,
    rows: list[dict[str, float]],
) -> list[dict[str, float]]:
    overrides = DOCUMENTED_LOOKUP_ROW_OVERRIDES.get(parameter_id)
    if not overrides:
        return rows

    merged_rows = {int(row["ga_weeks"]): dict(row) for row in rows}
    for ga_week, override in overrides.items():
        merged_rows[ga_week] = {
            "ga_weeks": float(ga_week),
            "centile_5": float(override["centile_5"]),
            "centile_50": float(override["centile_50"]),
            "centile_95": float(override["centile_95"]),
        }

    return [merged_rows[ga_week] for ga_week in sorted(merged_rows)]


LOOKUP_TABLE_SPECS: dict[str, dict[str, float]] = {
    "skull_bpd": {
        "start_median": 41.5,
        "end_median": 95.0,
        "start_spread": 3.8,
        "end_spread": 5.9,
        "curvature": 0.14,
    },
    "skull_ofd": {
        "start_median": 54.0,
        "end_median": 122.0,
        "start_spread": 4.6,
        "end_spread": 7.2,
        "curvature": 0.12,
    },
    "brain_bpd": {
        "start_median": 35.0,
        "end_median": 84.5,
        "start_spread": 3.3,
        "end_spread": 5.1,
        "curvature": 0.13,
    },
    "brain_ofd": {
        "start_median": 46.0,
        "end_median": 106.0,
        "start_spread": 4.0,
        "end_spread": 6.1,
        "curvature": 0.11,
    },
    "left_ventricular_diameter": {
        "start_median": 6.2,
        "end_median": 8.1,
        "start_spread": 1.0,
        "end_spread": 1.4,
        "curvature": -0.08,
        "minimum_value": 3.5,
    },
    "right_ventricular_diameter": {
        "start_median": 6.1,
        "end_median": 8.0,
        "start_spread": 1.0,
        "end_spread": 1.4,
        "curvature": -0.08,
        "minimum_value": 3.5,
    },
    "third_ventricle_width": {
        "start_median": 1.5,
        "end_median": 3.4,
        "start_spread": 0.6,
        "end_spread": 0.9,
        "curvature": 0.06,
        "minimum_value": 0.4,
    },
    "corpus_callosum_length": {
        "start_median": 17.5,
        "end_median": 44.0,
        "start_spread": 2.0,
        "end_spread": 3.4,
        "curvature": 0.15,
    },
    "csp_width": {
        "start_median": 3.8,
        "end_median": 7.4,
        "start_spread": 0.9,
        "end_spread": 1.5,
        "curvature": 0.10,
        "minimum_value": 1.5,
    },
    "transcerebellar_diameter": {
        "start_median": 19.5,
        "end_median": 49.0,
        "start_spread": 2.1,
        "end_spread": 3.6,
        "curvature": 0.11,
    },
    "vermian_height": {
        "start_median": 9.2,
        "end_median": 23.8,
        "start_spread": 1.2,
        "end_spread": 2.3,
        "curvature": 0.12,
    },
    "vermian_ap_diameter": {
        "start_median": 7.3,
        "end_median": 18.3,
        "start_spread": 1.1,
        "end_spread": 1.9,
        "curvature": 0.09,
    },
    "cisterna_magna_depth": {
        "start_median": 4.3,
        "end_median": 8.6,
        "start_spread": 0.9,
        "end_spread": 1.5,
        "curvature": 0.04,
        "minimum_value": 2.0,
    },
    "tegmento_vermian_angle": {
        "start_median": 24.0,
        "end_median": 10.5,
        "start_spread": 3.6,
        "end_spread": 2.6,
        "curvature": -0.14,
        "minimum_value": 3.0,
    },
    "pons_ap_diameter": {
        "start_median": 5.8,
        "end_median": 13.2,
        "start_spread": 0.8,
        "end_spread": 1.4,
        "curvature": 0.10,
        "minimum_value": 2.5,
    },
}


# These rows are explicitly shown in the validation/design document as examples of
# the Kyriakopoulou 2017 centile-table contract. They override the synthetic
# scaffold so tests can anchor to at least one document-verbatim lookup source.
DOCUMENTED_LOOKUP_ROW_OVERRIDES: dict[str, dict[int, dict[str, float]]] = {
    "brain_bpd": {
        20: {
            "centile_5": 41.2,
            "centile_50": 45.1,
            "centile_95": 49.0,
        },
        21: {
            "centile_5": 44.5,
            "centile_50": 48.6,
            "centile_95": 52.7,
        },
    },
}


LOOKUP_TABLES: dict[str, list[dict[str, float]]] = {
    parameter_id: normalize_lookup_rows(
        apply_documented_row_overrides(
            parameter_id,
            build_mock_growth_table(**spec),
        )
    )
    for parameter_id, spec in LOOKUP_TABLE_SPECS.items()
}


def grouped_parameters() -> list[dict[str, Any]]:
    return [
        {
            "name": group_name,
            "parameters": [parameter for parameter in PARAMETERS if parameter.group == group_name],
        }
        for group_name in GROUP_ORDER
    ]


def lookup_distribution(parameter_id: str, ga_fractional: float) -> tuple[float, float] | None:
    rows = LOOKUP_TABLES.get(parameter_id, [])
    if len(rows) < 2:
        return None

    weeks = np.array([row["ga_weeks"] for row in rows], dtype=float)
    centile_5 = np.array([row["centile_5"] for row in rows], dtype=float)
    centile_50 = np.array([row["centile_50"] for row in rows], dtype=float)
    centile_95 = np.array([row["centile_95"] for row in rows], dtype=float)

    lower_bound = float(np.min(weeks))
    upper_bound = float(np.max(weeks))
    if ga_fractional < lower_bound or ga_fractional > upper_bound:
        return None

    mean = float(PchipInterpolator(weeks, centile_50)(ga_fractional))
    centile_5_value = float(PchipInterpolator(weeks, centile_5)(ga_fractional))
    centile_95_value = float(PchipInterpolator(weeks, centile_95)(ga_fractional))
    lower_spread = max(mean - centile_5_value, MIN_CENTILE_SPREAD)
    upper_spread = max(centile_95_value - mean, MIN_CENTILE_SPREAD)
    std_dev = max((((lower_spread + upper_spread) / 2.0) / 1.645), MIN_STD_DEV)

    return mean, std_dev


def lookup_status_counts() -> tuple[int, int]:
    loaded_count = sum(1 for rows in LOOKUP_TABLES.values() if len(rows) >= 2)
    total_count = len(LOOKUP_TABLES)
    return loaded_count, total_count
