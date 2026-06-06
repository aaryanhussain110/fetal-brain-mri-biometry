# Fetal Brain MRI Biometry Calculator

Standalone FastAPI application for fetal brain MRI biometry entry, z-score/percentile calculation, structured report support, dynamic clinical warning cards, and local REF-corpus assisted AI consultation.

> **Educational prototype only.** This application is not a validated medical device and must not be used as a substitute for professional clinical judgment, institutional protocols, or independent verification of source literature.

The primary application entry point is `main.py`.

## Current Features

- Manual gestational age entry from 18w0d through 40w6d.
- Biometry inputs for global growth, ventricular system, midline structures, posterior fossa, and brainstem measurements.
- Real-time z-score and percentile calculation.
- Analytic Woitek-style posterior fossa curves for TDPF and CSA.
- PCHIP lookup-table interpolation for all table-backed parameters.
- Reference-shaped growth tables for weeks 18 through 40, with documented Kyriakopoulou example rows preserved for Brain BPD and safe interpolation for all table-backed parameters.
- Dynamic clinical warning cards for:
  - Mild-to-moderate ventriculomegaly.
  - Severe ventriculomegaly.
  - Asymmetric lateral ventricles.
  - Chiari II malformation / open neural tube defect pattern.
- Live structured preview panel and clipboard copy support.
- AI-RAG generated report card using local PDF extraction, cached TF-IDF retrieval, the `knowledge/` REF corpus, and Gemini when available.
- Literature Copilot chat that can answer patient-stat questions and paper/REF questions.
- Local fallback behavior when Gemini is unavailable, slow, or out of quota.

## RAG Alignment

This project follows the high-level workflow from the `sameerkhanna786/gemini_based_rag` teaching project without copying its full package structure:

- Local PDFs are stored as stable `REF_###__title.pdf` corpus files under `knowledge/`.
- PDF text is extracted locally with `pypdf`.
- Retrieved context is ranked with a lightweight local `scikit-learn` TF-IDF index built and cached at runtime.
- Retrieved chunks are labeled as `[C1]`, `[C2]`, etc. so Gemini can ground factual claims in the supplied context.
- Gemini is used only for final answer/report synthesis when `GEMINI_API_KEY` is configured.
- The app falls back to deterministic local calculator and REF-corpus logic when Gemini is unavailable, slow, or out of quota.
- API keys and local environment files are not committed.

## Sample Data Provenance

The `SAMPLE: NORMAL` and `SAMPLE: FLAGGED` buttons are reference-grounded demonstration profiles, not de-identified patient cases.

- `SAMPLE: NORMAL` is a representative 21w0d profile aligned to the calculator's normative reference curves. The Brain BPD anchor preserves the documented Kyriakopoulou-style 21-week 50th centile example used by the test suite.
- `SAMPLE: FLAGGED` is a constructed 24w0d stress-test profile designed to trigger clinically relevant warning pathways: severe ventriculomegaly at atrial diameter >= 15 mm and the Woitek-style posterior-fossa TDPF/CSA abnormal geometry branch.
- This is intentional: the source papers generally provide aggregate centiles, equations, thresholds, and outcome associations rather than complete reusable raw patient rows. The calculator therefore uses published reference logic and transparent rule-based samples rather than pretending to contain literal extracted patient data.

## Run Locally

Install dependencies once:

```powershell
python -m pip install -r requirements.txt
```

Optional, for live Gemini responses:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

Optional Gemini timeout settings:

```powershell
$env:GEMINI_REPORT_TIMEOUT_SECONDS="30"
$env:GEMINI_CHAT_TIMEOUT_SECONDS="20"
$env:GEMINI_PAPER_CHAT_TIMEOUT_SECONDS="45"
```

These defaults give Gemini enough time to produce grounded responses while preserving the local calculator and paper-corpus fallback if Gemini is unavailable, quota-limited, or exceeds the configured timeout.

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
