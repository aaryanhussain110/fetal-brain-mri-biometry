# Fetal Brain MRI Biometry Calculator

Standalone FastAPI application for fetal brain MRI biometry entry, z-score/percentile calculation, structured report support, dynamic clinical warning cards, and local REF-corpus assisted AI consultation.

The primary application entry point is `main.py`.

## Current Features

- Manual gestational age entry from 18w0d through 40w6d.
- Biometry inputs for global growth, ventricular system, midline structures, posterior fossa, and brainstem measurements.
- Real-time z-score and percentile calculation.
- Analytic Woitek-style posterior fossa curves for TDPF and CSA.
- PCHIP lookup-table interpolation for all table-backed parameters.
- Complete mock growth tables for weeks 18 through 40, with documented Kyriakopoulou example rows preserved for Brain BPD.
- Dynamic clinical warning cards for:
  - Mild-to-moderate ventriculomegaly.
  - Severe ventriculomegaly.
  - Asymmetric lateral ventricles.
  - Chiari II malformation / open neural tube defect pattern.
- Live structured preview panel and clipboard copy support.
- AI-RAG generated report card using the local `knowledge/` REF corpus and Gemini when available.
- Literature Copilot chat that can answer patient-stat questions and paper/REF questions.
- Local fallback behavior when Gemini is unavailable, slow, or out of quota.

## Run Locally

Install dependencies once:

```powershell
python -m pip install -r requirements.txt
```

Optional, for live Gemini responses:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

Start the local server:

```powershell
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Important Files

- `main.py`: primary FastAPI app, routes, calculator orchestration, report generation, warning cards, Gemini/Copilot logic.
- `data_reference.py`: parameter definitions, analytic curve metadata, lookup table generation, safe interpolation logic.
- `templates/index.html`: outer application shell and client-side UI behavior.
- `templates/partials/workspace.html`: HTMX-updated calculator workspace, report panel, and warning-card layout.
- `knowledge/`: local medical REF corpus used by report generation and Copilot retrieval.
- `tests/test_calculator.py`: current numerical and workflow regression suite.
- `app/`: legacy compatibility package retained in the workspace, but not the primary documented entry point.

## Corpus Attribution

The `knowledge/` folder is included for course/demo RAG grounding. It is based on the instructor-provided public `sameerkhanna786/gemini_based_rag` corpus layout, where the PDF corpus is published under `papers/corpus`.

The PDFs are used here as a local retrieval corpus for educational radiology decision-support prototyping. If this project is reused outside the course/demo context, verify each paper's redistribution terms and replace the corpus with appropriately licensed or institutionally approved reference material.

## Testing

Run the current unit suite:

```powershell
python -m unittest tests\test_calculator.py
```

Expected result:

```text
Ran 9 tests
OK
```

## Privacy And Safety

- The app is designed for local workstation use.
- Measurements are held in browser/session form state and server request context only.
- Refreshing or clearing the workspace removes entered values.
- The AI report and Copilot are decision-support tools, not autonomous clinical decision makers.
- Final clinical interpretation remains the responsibility of the radiologist and local care team.
