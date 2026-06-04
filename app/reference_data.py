from __future__ import annotations

import json
from pathlib import Path

from app.domain import LookupPoint, LookupTable, MeasurementDefinition


DATA_FILE = Path(__file__).parent / "data" / "reference_tables.json"


GROUP_ORDER = (
    "Global Growth",
    "Ventricular System",
    "Midline Structures",
    "Posterior Fossa",
    "Brainstem",
)


MEASUREMENT_DEFINITIONS: tuple[MeasurementDefinition, ...] = (
    MeasurementDefinition(
        parameter_id="skull_bpd",
        label="Skull BPD",
        unit="mm",
        group="Global Growth",
        description="Outer biparietal diameter measured across the calvarium.",
        method="lookup",
        source="Lookup table import required",
    ),
    MeasurementDefinition(
        parameter_id="skull_ofd",
        label="Skull OFD",
        unit="mm",
        group="Global Growth",
        description="Outer occipitofrontal diameter measured across the calvarium.",
        method="lookup",
        source="Lookup table import required",
    ),
    MeasurementDefinition(
        parameter_id="brain_bpd",
        label="Brain BPD",
        unit="mm",
        group="Global Growth",
        description="Inner cerebral biparietal diameter.",
        method="lookup",
        source="Lookup table import required",
    ),
    MeasurementDefinition(
        parameter_id="brain_ofd",
        label="Brain OFD",
        unit="mm",
        group="Global Growth",
        description="Inner cerebral occipitofrontal diameter.",
        method="lookup",
        source="Lookup table import required",
    ),
    MeasurementDefinition(
        parameter_id="left_ventricular_diameter",
        label="Left Ventricular Diameter",
        unit="mm",
        group="Ventricular System",
        description="Atrial width of the left lateral ventricle.",
        method="lookup",
        source="Lookup table import required",
    ),
    MeasurementDefinition(
        parameter_id="right_ventricular_diameter",
        label="Right Ventricular Diameter",
        unit="mm",
        group="Ventricular System",
        description="Atrial width of the right lateral ventricle.",
        method="lookup",
        source="Lookup table import required",
    ),
    MeasurementDefinition(
        parameter_id="third_ventricle_width",
        label="Third Ventricle Width",
        unit="mm",
        group="Ventricular System",
        description="Maximum transverse width of the third ventricle.",
        method="lookup",
        source="Lookup table import required",
    ),
    MeasurementDefinition(
        parameter_id="corpus_callosum_length",
        label="Corpus Callosum Length",
        unit="mm",
        group="Midline Structures",
        description="Sagittal corpus callosum length.",
        method="lookup",
        source="Lookup table import required",
    ),
    MeasurementDefinition(
        parameter_id="csp_width",
        label="CSP Width",
        unit="mm",
        group="Midline Structures",
        description="Maximum cavum septum pellucidum width.",
        method="lookup",
        source="Lookup table import required",
    ),
    MeasurementDefinition(
        parameter_id="transcerebellar_diameter",
        label="Transcerebellar Diameter",
        unit="mm",
        group="Posterior Fossa",
        description="Maximum transverse cerebellar diameter.",
        method="lookup",
        source="Lookup table import required",
    ),
    MeasurementDefinition(
        parameter_id="vermian_height",
        label="Vermian Height",
        unit="mm",
        group="Posterior Fossa",
        description="Superior-inferior vermian dimension.",
        method="lookup",
        source="Lookup table import required",
    ),
    MeasurementDefinition(
        parameter_id="vermian_ap_diameter",
        label="Vermian AP Diameter",
        unit="mm",
        group="Posterior Fossa",
        description="Anterior-posterior vermian dimension.",
        method="lookup",
        source="Lookup table import required",
    ),
    MeasurementDefinition(
        parameter_id="cisterna_magna_depth",
        label="Cisterna Magna Depth",
        unit="mm",
        group="Posterior Fossa",
        description="Anteroposterior depth of the cisterna magna.",
        method="lookup",
        source="Lookup table import required",
    ),
    MeasurementDefinition(
        parameter_id="tegmento_vermian_angle",
        label="Tegmento-Vermian Angle",
        unit="degrees",
        group="Posterior Fossa",
        description="Angle formed by the dorsal brainstem and ventral vermis.",
        method="lookup",
        source="Lookup table import required",
    ),
    MeasurementDefinition(
        parameter_id="tdpf",
        label="Posterior Fossa Diameter (TDPF)",
        unit="mm",
        group="Posterior Fossa",
        description="Maximum transverse diameter of the posterior fossa.",
        method="analytic",
        source="Analytic regression formula",
    ),
    MeasurementDefinition(
        parameter_id="csa",
        label="Clivus-Supraocciput Angle (CSA)",
        unit="degrees",
        group="Posterior Fossa",
        description="Angle subtended by the clivus and supraocciput.",
        method="analytic",
        source="Analytic regression formula",
    ),
    MeasurementDefinition(
        parameter_id="pons_ap_diameter",
        label="Pons AP Diameter",
        unit="mm",
        group="Brainstem",
        description="Anterior-posterior pons diameter.",
        method="lookup",
        source="Lookup table import required",
    ),
)


DEFINITION_BY_ID = {item.parameter_id: item for item in MEASUREMENT_DEFINITIONS}


def build_fallback_lookup_tables() -> dict[str, LookupTable]:
    try:
        from data_reference import LOOKUP_TABLES as ROOT_LOOKUP_TABLES
    except ModuleNotFoundError:
        return {}

    tables: dict[str, LookupTable] = {}
    for parameter_id, rows in ROOT_LOOKUP_TABLES.items():
        data_points = tuple(
            LookupPoint(
                ga_weeks=int(row["ga_weeks"]),
                centile_5=float(row["centile_5"]),
                centile_50=float(row["centile_50"]),
                centile_95=float(row["centile_95"]),
            )
            for row in rows
        )
        tables[parameter_id] = LookupTable(
            parameter_id=parameter_id,
            source="Fallback root lookup table import",
            data=data_points,
        )

    return tables


def load_lookup_tables() -> dict[str, LookupTable]:
    if not DATA_FILE.exists():
        return build_fallback_lookup_tables()

    raw_rows = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    tables: dict[str, LookupTable] = {}

    for raw_table in raw_rows:
        data_points = tuple(
            LookupPoint(
                ga_weeks=int(point["ga_weeks"]),
                centile_5=float(point["centile_5"]),
                centile_50=float(point["centile_50"]),
                centile_95=float(point["centile_95"]),
            )
            for point in raw_table.get("data", [])
        )

        tables[raw_table["parameter_id"]] = LookupTable(
            parameter_id=raw_table["parameter_id"],
            source=raw_table.get("source", "Unknown source"),
            data=data_points,
        )

    loaded_any = any(len(table.data) >= 2 for table in tables.values())
    return tables if loaded_any else build_fallback_lookup_tables()

DIAGNOSTIC_LITERATURE_REGISTRY = {
    "mild_moderate_ventriculomegaly": {
        "condition": "Mild-to-Moderate Ventriculomegaly",
        "trigger": "Atrial diameter >= 10 mm and < 15 mm",
        "likelihood": "~8%",
        "rationale": "Neurodevelopmental delay risk is low (~7.9%) in isolated mild VM cases.",
        "citation": "Pagani G, et al. Ultrasound Obstet Gynecol. 2014;44(3):254-260.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/?term=Pagani+2014+Ultrasound+Obstet+Gynecol+44+3+254-260"
    },
    "severe_ventriculomegaly": {
        "condition": "Severe Fetal Ventriculomegaly",
        "trigger": "Atrial diameter >= 15 mm",
        "likelihood": "~10-20% Isolated",
        "rationale": "Marker of significant underlying brain pathology; high association with aqueductal stenosis.",
        "citation": "Giorgione V, et al. Prenat Diagn. 2022;42(13):1674-1681.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/?term=Giorgione+2022+Prenat+Diagn+42+13+1674-1681"
    },
    "asymmetric_ventricles": {
        "condition": "Asymmetric Atrial Diameters",
        "trigger": "Right vs Left ventricle difference > 2 mm",
        "likelihood": "~37-46% Progression Risk",
        "rationale": "Often benign if isolated, but carries high progression correlation to true ventriculomegaly.",
        "citation": "Barzilay E, et al. AJNR Am J Neuroradiol. 2017;38(2):371-375.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/?term=Barzilay+2017+AJNR+38+2+371-375"
    },
    "chiari_ii_ontd": {
        "condition": "Chiari II Malformation / Open Neural Tube Defect",
        "trigger": "TDPF z < -2.0 AND CSA z < -2.0 AND posterior probability > 0.5",
        "likelihood": "~85-90%",
        "rationale": "Small posterior fossa configuration with acute clivus angle is the definitive signature of spinal dysraphism.",
        "citation": "Woitek R, Prayer D, Weber M, et al. PLOS One. 2014;9(11):e112585.",
        "url": "https://doi.org/10.1371/journal.pone.0112585"
    }
}
