import numpy as np
from scipy.interpolate import PchipInterpolator
import scipy.stats as stats

DIAGNOSTIC_LITERATURE_REGISTRY = {
    "mild_moderate_ventriculomegaly": {
        "condition": "Mild-to-Moderate Ventriculomegaly",
        "trigger": "Atrial diameter >= 10 mm and < 15 mm",
        "likelihood": "~8%",
        "rationale": "Overall neurodevelopmental delay risk is low (~7.9%) in isolated mild VM.",
        "citation": "Pagani G, et al. Ultrasound Obstet Gynecol. 2014;44(3):254-260.",
        "url": "https://nih.gov"
    },
    "severe_ventriculomegaly": {
        "condition": "Severe Fetal Ventriculomegaly",
        "trigger": "Atrial diameter >= 15 mm",
        "likelihood": "~10-20% Isolated",
        "rationale": "Marker of significant underlying brain pathology, often associated with aqueductal stenosis (~20%).",
        "citation": "Giorgione V, et al. Prenat Diagn. 2022;42(13):1674-1681.",
        "url": "https://nih.gov"
    },
    "asymmetric_ventricles": {
        "condition": "Asymmetric Atrial Diameters",
        "trigger": "Right vs Left ventricle difference > 2 mm",
        "likelihood": "~37-46% Progression Risk",
        "rationale": "Often a benign variant if isolated, but a significant percentage of cases progress to ventriculomegaly.",
        "citation": "Barzilay E, et al. AJNR Am J Neuroradiol. 2017;38(2):371-375.",
        "url": "https://nih.gov"
    },
    "chiari_ii_ontd": {
        "condition": "Chiari II Malformation / Open Neural Tube Defect",
        "trigger": "TDPF z < -2.0 AND CSA z < -2.0",
        "likelihood": "~85-90%",
        "rationale": "A small posterior fossa with an acute clivus angle is the definitive cranial signature of open spinal dysraphism.",
        "citation": "Woitek R, Prayer D, Weber M, et al. PLOS One. 2014;9(11):e112585.",
        "url": "https://nih.gov"
    }
}

def calculate_fractional_ga(weeks: int, days: int) -> float:
    """Calculates continuous decimal gestational age (GA = Weeks + Days/7)."""
    return float(weeks + (days / 7.0))

def calculate_analytic_posterior_fossa(ga: float, tdpf_val: float = None, csa_val: float = None):
    """
    Computes precise analytical OLS quadratic mean and linear SD 
    derived from Woitek 2014 control cohorts (Section 6.5.2).
    """
    outputs = {}
    if tdpf_val is not None:
        mu_tdpf = -0.01307 * (ga ** 2) + 2.55571 * ga - 21.71
        sigma_tdpf = 0.06716 * ga + 0.547
        z_tdpf = (tdpf_val - mu_tdpf) / sigma_tdpf
        p_tdpf = stats.norm.cdf(z_tdpf) * 100
        outputs['tdpf'] = {"z": round(z_tdpf, 2), "p": round(p_tdpf, 1)}

    if csa_val is not None:
        mu_csa = -0.04767 * (ga ** 2) + 4.20404 * ga + 1.73
        sigma_csa = 0.01814 * ga + 5.821
        z_csa = (csa_val - mu_csa) / sigma_csa
        p_csa = stats.norm.cdf(z_csa) * 100
        outputs['csa'] = {"z": round(z_csa, 2), "p": round(p_csa, 1)}

    return outputs

def run_kyriakopoulou_fallback_spline(ga_weeks: float, measurement: float):
    """
    Executes monotone cubic spline interpolation fallback using standard tabular centiles 
    modeled on Kyriakopoulou 2017 schema (Section 4.2 schema specification).
    """
    static_table = [
        {"ga": 19.0, "c5": 38.0, "c50": 42.0, "c95": 46.0},
        {"ga": 20.0, "c5": 41.2, "c50": 45.1, "c95": 49.0},
        {"ga": 21.0, "c5": 44.5, "c50": 48.6, "c95": 52.7}
    ]
    
    ga_axis = [row["ga"] for row in static_table]
    c50_axis = [row["c50"] for row in static_table]
    c5_axis = [row["c5"] for row in static_table]
    c95_axis = [row["c95"] for row in static_table]
    
    interp_mu = PchipInterpolator(ga_axis, c50_axis)
    interp_c5 = PchipInterpolator(ga_axis, c5_axis)
    interp_c95 = PchipInterpolator(ga_axis, c95_axis)
    
    mu = float(interp_mu(ga_weeks))
    val_c5 = float(interp_c5(ga_weeks))
    val_c95 = float(interp_c95(ga_weeks))
    
    sigma = ((mu - val_c5) / 1.645 + (val_c95 - mu) / 1.645) / 2.0
    
    z_score = (measurement - mu) / sigma
    percentile = stats.norm.cdf(z_score) * 100
    
    return {"z": round(z_score, 2), "p": round(percentile, 1)}

def evaluate_differential_triggers(left_v: float = None, right_v: float = None, tdpf_z: float = None, csa_z: float = None) -> list:
    """Evaluates biometric limits and populates active trigger flags."""
    active_keys = []
    
    if left_v and right_v:
        max_atrial = max(left_v, right_v)
        if 10.0 <= max_atrial < 15.0:
            active_keys.append("mild_moderate_ventriculomegaly")
        elif max_atrial >= 15.0:
            active_keys.append("severe_ventriculomegaly")
            
        if abs(right_v - left_v) > 2.0:
            active_keys.append("asymmetric_ventricles")
            
    if tdpf_z is not None and csa_z is not None:
        if tdpf_z < -2.0 and csa_z < -2.0:
            active_keys.append("chiari_ii_ontd")
            
    return active_keys
