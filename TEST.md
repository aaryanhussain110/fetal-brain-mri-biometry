# TEST.md

## Test Command

Run:

```powershell
python -m unittest tests\test_calculator.py
```

Current expected result:

```text
Ran 9 tests
OK
```

## Unit Test Coverage

The current test suite verifies the following 9 cases.

1. TDPF analytic distribution at 24w0d.
   - Expected mean: `32.09872`.
   - Expected standard deviation: `2.15884`.

2. CSA analytic distribution at 24w0d.
   - Expected mean: `75.16904`.
   - Expected standard deviation: `6.25636`.

3. Worked posterior fossa z-score example.
   - TDPF input: `24.0 mm` at 24w0d.
   - Expected z-score: approximately `-3.751422`.
   - CSA input: `55.0 deg` at 24w0d.
   - Expected z-score: approximately `-3.223766`.
   - Percentiles must match the normal CDF of the calculated z-scores.

4. Chiari II / open NTD joint classifier.
   - Uses the worked TDPF and CSA example.
   - Expected Chiari joint probability: greater than `0.99`.

5. Flagged sample warning-card behavior.
   - GA: 24w0d.
   - Left ventricular diameter: `16.0 mm`.
   - Right ventricular diameter: `15.2 mm`.
   - TDPF: `24.0 mm`.
   - CSA: `55.0 deg`.
   - Expected warning cards: Severe Ventriculomegaly and Chiari II Malformation / Open NTD.

6. Normal sample warning-card behavior.
   - GA: 21w0d.
   - Brain BPD: `48.6 mm`.
   - TDPF: `26.1960 mm`.
   - CSA: `68.9924 deg`.
   - Expected warning cards: none.

7. Severe ventriculomegaly threshold inclusivity.
   - Left ventricular diameter: `15.0 mm`.
   - Expected warning card: Severe Ventriculomegaly.

8. Documented Brain BPD lookup rows are preserved.
   - 20w row: 5th `41.2`, 50th `45.1`, 95th `49.0`.
   - 21w row: 5th `44.5`, 50th `48.6`, 95th `52.7`.

9. Lookup distributions are safe for all table-backed parameters.
   - Every lookup table must return a non-null distribution at 24w0d.
   - Mean must be positive.
   - Standard deviation must be positive.

## Manual Smoke Checks

These are useful after UI or Copilot changes.

1. Load the home page and confirm no warning cards appear before data entry.
2. Click `SAMPLE: NORMAL`; all rows should populate and no diagnostic warning card should trigger.
3. Click `SAMPLE: FLAGGED`; severe ventriculomegaly and posterior fossa warning cards should appear.
4. Confirm the Generate Report button stays disabled until every measurement field has a value.
5. Generate a report from a completed sample and confirm the AI-RAG report card appears below the calculator.
6. Ask Copilot: `What are the patient diagnoses?`
7. Ask Copilot: `Explain REF_037.`
8. Ask Copilot: `Explain paper 36.`
   - Expected response: `REF_036` is not present in the local knowledge corpus.
