# TASK.md

## Current Operational Task List

The project is currently in a functional validation and polish phase. Core calculator scaffolding, math routines, sample loading, dynamic warning cards, report generation, and Copilot fallback behavior are implemented.

## Implemented Inputs

The UI currently accepts:

1. Gestational age weeks and days.
2. Skull BPD and Skull OFD.
3. Brain BPD and Brain OFD.
4. Left and right lateral ventricular diameters.
5. Third ventricle width.
6. Corpus callosum length.
7. CSP width.
8. Transcerebellar diameter.
9. Vermian height and Vermian AP diameter.
10. Pons AP diameter.
11. Cisterna magna depth.
12. Tegmento-vermian angle.
13. TDPF.
14. CSA.
15. Imaging quality selectors for field strength, motion artifact, and reference profile.

## Implemented Outputs

The application currently produces:

1. Real-time parameter rows with z-score, percentile, and status labels.
2. Distribution marker UI for measurement position.
3. Dynamic clinical warning cards.
4. Applicable REF paper chips on warning cards.
5. Structured plain-text preview for clipboard copy.
6. AI-RAG generated report card after all measurements are complete.
7. Literature Copilot chat for patient-context and paper-corpus questions.
8. Local fallback answers when Gemini is unavailable or quota-limited.

## Current Warning Logic

The dynamic warning section currently supports:

1. Mild-to-moderate ventriculomegaly.
2. Severe ventriculomegaly.
3. Asymmetric lateral ventricles.
4. Chiari II malformation / open neural tube defect pattern.

## Current Validation Work

The active test suite is:

```powershell
python -m unittest tests\test_calculator.py
```

Expected status:

```text
Ran 9 tests
OK
```

## Known Documentation Notes

- The primary app entry point is `main.py`.
- Generated lookup tables are realistic reference-shaped growth curves with documented Brain BPD overrides, not a full verbatim reproduction of every source-paper table.
- Sample profiles are reference-grounded demonstration and stress-test profiles, not literal patient cases extracted from the papers.
- The AI layer is assistive and source-grounded, but final clinical report approval remains with the radiologist.
