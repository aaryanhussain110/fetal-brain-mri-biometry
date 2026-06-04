# SPEC.md

## 1. Current Product Scope

The Fetal Brain MRI Biometry Calculator is a local FastAPI and HTMX web application for manual fetal brain MRI biometry entry. It computes z-scores and percentiles, renders a structured report preview, activates evidence-linked warning cards, and provides an AI-RAG report/Copilot workflow grounded in the local `knowledge/` REF corpus.

The current live entry point is `main.py`.

## 2. Technical Stack

- Backend: FastAPI, Python 3.11+
- Templates: Jinja2
- UI updates: HTMX with a local JavaScript fallback path
- Styling: Tailwind CSS via CDN
- Math: NumPy, SciPy PCHIP interpolation, normal CDF
- AI/RAG: `google-genai` using `gemini-2.5-flash` when configured
- Corpus reading: `pathlib` and `pypdf`
- Tests: Python `unittest`

## 3. Supported Measurements

The current parameter set contains 17 biometric measurements.

- Global growth: Skull BPD, Skull OFD, Brain BPD, Brain OFD.
- Ventricular system: Left ventricular diameter, Right ventricular diameter, Third ventricle width.
- Midline structures: Corpus callosum length, CSP width.
- Posterior fossa: Transcerebellar diameter, Vermian height, Vermian AP diameter, Cisterna magna depth, Tegmento-vermian angle, TDPF, CSA.
- Brainstem: Pons AP diameter.

## 4. Calculation Rules

Fractional gestational age:

```text
GA_fractional = weeks + days / 7
```

Z-score:

```text
z = (observed_value - mean) / standard_deviation
```

Percentile:

```text
percentile = normal_cdf(z) * 100
```

### Analytic Curves

TDPF and CSA use analytic posterior fossa formulas.

TDPF:

```text
mean = -0.01307 * GA^2 + 2.55571 * GA - 21.71
std_dev = 0.06716 * GA + 0.547
```

CSA:

```text
mean = -0.04767 * GA^2 + 4.20404 * GA + 1.73
std_dev = 0.01814 * GA + 5.821
```

Current verified 24w0d anchors:

- TDPF mean `32.09872`, standard deviation `2.15884`.
- CSA mean `75.16904`, standard deviation `6.25636`.
- TDPF input `24.0 mm` gives z-score approximately `-3.7514`.
- CSA input `55.0 deg` gives z-score approximately `-3.2238`.

### Lookup Tables

All other lookup-backed parameters use generated 18 through 40 week centile rows in `data_reference.py`.

The generated tables preserve the documented Brain BPD rows:

```json
{"ga_weeks": 20, "centile_5": 41.2, "centile_50": 45.1, "centile_95": 49.0}
{"ga_weeks": 21, "centile_5": 44.5, "centile_50": 48.6, "centile_95": 52.7}
```

PCHIP interpolation resolves the 5th, 50th, and 95th centiles at fractional GA. The 50th centile is treated as the mean. Standard deviation is recovered from the average distance to the 5th and 95th centiles using `1.645` as the normal-distribution 5th/95th z-distance.

## 5. Clinical Warning Cards

The current dynamic warning section renders four card families.

- Mild-to-moderate ventriculomegaly: left or right atrial diameter `>= 10.0 mm` and `< 15.0 mm`.
- Severe ventriculomegaly: left or right atrial diameter `>= 15.0 mm`.
- Asymmetric lateral ventricles: absolute left-right atrial difference `> 2.0 mm`.
- Chiari II malformation / open NTD pattern: TDPF z-score `< -2.0`, CSA z-score `< -2.0`, and Chiari joint probability `> 0.5`.

Third ventricle width is calculated and reported, but there is not currently a separate third-ventricle warning card in the live code.

## 6. Reference Paper Binding

Warning cards pass applicable REF papers into the template:

- Mild-to-moderate ventriculomegaly: `REF_015`, `REF_019`.
- Severe ventriculomegaly: `REF_016`, `REF_017`.
- Asymmetric lateral ventricles: `REF_018`, `REF_019`.
- Chiari II / open NTD: `REF_046`, `REF_049`.

The local `knowledge/` folder currently contains 44 REF files. REF numbering has intentional gaps; for example, `REF_036` is not present.

## 7. AI-RAG Report And Copilot

The AI report button posts to:

```text
POST /generate-report
```

The report button is disabled until all required measurements are filled.

The Copilot chat posts to:

```text
POST /chat-consult
```

Copilot behavior:

- Patient-stat, measurement, and diagnosis questions are answered from the current hidden patient summary.
- Paper questions are routed through query-specific REF retrieval.
- Missing paper numbers return a clear missing-corpus response instead of borrowing from patient context.
- Gemini is attempted when configured and quota is available.
- Local fallback answers are used when Gemini is unavailable, out of quota, or too slow.

## 8. Acceptance Status

The current regression suite is `tests/test_calculator.py`.

Current status:

```text
9 tests passing
```
