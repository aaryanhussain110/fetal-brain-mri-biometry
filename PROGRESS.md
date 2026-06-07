# PROGRESS.md

## System Status

- [x] Primary FastAPI app implemented in `main.py`.
- [x] HTMX/Jinja workspace implemented.
- [x] Calculation engine implemented for analytic and lookup-backed parameters.
- [x] Reference-shaped growth lookup tables populated for weeks 18 through 40.
- [x] Documented Brain BPD Kyriakopoulou example rows preserved.
- [x] Dynamic warning cards implemented.
- [x] Sample Normal and Sample Flagged workflows implemented.
- [x] AI-RAG report endpoint implemented.
- [x] Literature Copilot endpoint implemented.
- [x] Local REF-corpus fallback behavior implemented.
- [x] Current unit suite green.

## Current Test Status

```text
tests/test_calculator.py: 9/9 passing
```

## Completed Milestones

### Calculator Core

- Fractional gestational age helper implemented.
- Analytic TDPF and CSA formulas implemented.
- PCHIP lookup interpolation implemented.
- Safe non-zero standard deviation handling implemented.
- Status labels and percentile calculation implemented.

### Clinical Warning Engine

- Mild-to-moderate ventriculomegaly card implemented.
- Severe ventriculomegaly card implemented.
- Asymmetric lateral ventricles card implemented.
- Chiari II / open NTD posterior fossa card implemented.
- Applicable REF paper chips added to warning cards.

### UI Workflow

- Sticky clinical header implemented.
- Sample Normal and Sample Flagged controls implemented.
- Clear workspace control implemented.
- Live preview and copy workflow implemented.
- Generate Report button gated until all measurements are complete.
- Emergency alert banner visually polished.

### AI-RAG And Copilot

- `POST /generate-report` implemented.
- `POST /chat-consult` implemented.
- Local `knowledge/` REF corpus integrated.
- Local PDF extraction and cached TF-IDF retrieval implemented for report/Copilot grounding.
- Gemini attempted when API key, package, and quota are available.
- Gemini report and chat timeouts are deployment-configurable, with longer defaults to avoid unnecessary fallback responses.
- Local fallback handles Gemini timeout, quota, or missing package.
- Patient-stat and diagnosis questions are answered from the current patient summary.
- Paper questions route to REF-specific retrieval.
- Missing REF numbers return a clear missing-corpus answer.

### Documentation

- README, SPEC, TASK, TEST, and PROGRESS updated to match the current project state.

## Notes

- The primary documented server command is `uvicorn main:app --reload`.
- The REF corpus has numbering gaps. For example, `REF_036` is not present.
- Current lookup tables are realistic generated reference curves with documented row overrides, not a complete verbatim reproduction of every source table.
- Sample profiles are reference-grounded normal and abnormal reference profiles, not literal patient cases extracted from the papers.
