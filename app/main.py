from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import main as platform_main
from app.domain import WorkspaceState
from app.reference_data import MEASUREMENT_DEFINITIONS, load_lookup_tables
import os
from fastapi import Form
from fastapi.responses import HTMLResponse
from google import genai

client = genai.Client()



APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
templates = Jinja2Templates(directory=str(PROJECT_DIR / "templates"))
lookup_tables = load_lookup_tables()

app = FastAPI(title="Fetal Brain MRI Biometry Calculator")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
@app.post("/generate-report", response_class=HTMLResponse)
async def generate_report(request: Request):
    form_data = await request.form()
    ga_weeks = form_data.get("ga_weeks", "22")
    ga_days = form_data.get("ga_days", "0")
    left_ventricle = form_data.get("va_left", "").strip()
    right_ventricle = form_data.get("va_right", "").strip()
    
    lv_val = float(left_ventricle) if left_ventricle else 8.5
    rv_val = float(right_ventricle) if right_ventricle else 8.5
    max_ventricle = max(lv_val, rv_val)
    ventricle_diff = abs(lv_val - rv_val)

    retrieved_context = ""
    if max_ventricle >= 15.0:
        retrieved_context += "Giorgione 2022 Guidelines: Atrial widths >= 15.0 mm classify as Severe Ventriculomegaly. Mandate survey, fetal echo, infection panel, and neurosurgical counseling.\n"
    elif 10.0 <= max_ventricle <= 14.9:
        retrieved_context += "Pagani 2014 Guidelines: Atrial width between 10.0 mm and 14.9 mm is Mild-to-Moderate Ventriculomegaly. Schedule tracking scans every 2-3 weeks and genetic counseling.\n"
    if ventricle_diff > 2.0:
        retrieved_context += "Barzilay 2017 Guidelines: Hemispheric diameter asymmetry > 2.0 mm requires independent compliance logging to rule out localized obstructive mechanisms.\n"
    if not retrieved_context:
        retrieved_context = "Standard Fetal Neuro-imaging Parameters: Atrial dimensions under 10.0 mm are considered within standard expected normative boundaries."

    rag_prompt = f"""
    You are an expert pediatric neuroradiologist. Generate a professional structured report based strictly on the provided medical context rules.
    [CLINICAL GUIDELINES CORPUS]
    {retrieved_context}
    [CASE SPECIFICATIONS]
    - Gestational Age: {ga_weeks} Weeks, {ga_days} Days
    - Left Lateral Ventricle Atrial Diameter: {lv_val} mm
    - Right Lateral Ventricle Atrial Diameter: {rv_val} mm
    Format the response using clean, raw HTML snippet tags containing exactly three bold headers:
    1. <strong>METRICS SUMMARY LOG</strong>
    2. <strong>GROUNDED RESEARCH DIAGNOSIS</strong> (Cite papers if broken thresholds occur)
    3. <strong>MANDATED RADIOLOGIST CLINICAL NEXT STEPS</strong> (Provide the exact actionable advice from the context)
    Do NOT include markdown backticks. Keep it clean and dense.
    """
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=rag_prompt)
        ai_text = response.text
    except Exception as e:
        ai_text = f"<p class='text-red-600 font-semibold'>RAG Connection Error: {str(e)}</p>"

    return f"""
    <div id="final-radiology-report" class="p-6 bg-emerald-50 border-l-4 border-emerald-600 rounded-xl shadow mt-6 text-slate-800 animate-fade-in">
        <div class="flex items-center justify-between mb-4 border-b pb-2 border-emerald-200">
            <h3 class="text-md font-bold tracking-wide uppercase flex items-center gap-1.5 text-emerald-900">🟢 AI-RAG Verified Assistant Report</h3>
            <button onclick="window.print()" class="bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold py-1 px-3 border border-slate-300 rounded-lg shadow-sm transition">🖨️ Save as PDF / Print Report</button>
        </div>
        <div class="text-sm space-y-4 leading-relaxed">{ai_text}</div>
    </div>
    """

def default_measurements() -> dict[str, float | None]:
    return {definition.parameter_id: None for definition in MEASUREMENT_DEFINITIONS}


def parse_workspace_state(form_data) -> WorkspaceState:
    measurements = default_measurements()
    for definition in MEASUREMENT_DEFINITIONS:
        measurements[definition.parameter_id] = parse_optional_float(form_data.get(definition.parameter_id))

    return WorkspaceState(
        ga_weeks=parse_int(form_data.get("ga_weeks"), 22, minimum=18, maximum=40),
        ga_days=parse_int(form_data.get("ga_days"), 0, minimum=0, maximum=6),
        field_strength=form_data.get("field_strength", "1.5T"),
        motion_artifact=form_data.get("motion_artifact", "None"),
        reference_profile=form_data.get("reference_profile", "Standard curves"),
        measurements=measurements,
    )


def parse_optional_float(raw_value) -> float | None:
    if raw_value is None:
        return None

    value = str(raw_value).strip()
    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def parse_int(raw_value, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def render_workspace(request: Request, state: WorkspaceState) -> HTMLResponse:
    context = platform_main.build_page_context(form_values_from_state(state))
    template_name = (
        "partials/workspace.html"
        if request.headers.get("HX-Request") == "true"
        or request.headers.get("X-Requested-With") == "fetch"
        else "index.html"
    )
    return templates.TemplateResponse(
        request=request,
        name=template_name,
        context=context,
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return render_workspace(request, default_state())


@app.post("/calculate", response_class=HTMLResponse)
async def calculate(request: Request) -> HTMLResponse:
    form = await request.form()
    return render_workspace(request, parse_workspace_state(form))


@app.post("/workspace", response_class=HTMLResponse)
async def workspace(request: Request) -> HTMLResponse:
    return await calculate(request)


def default_state() -> WorkspaceState:
    return WorkspaceState(
        ga_weeks=22,
        ga_days=0,
        field_strength="1.5T",
        motion_artifact="None",
        reference_profile="Standard curves",
        measurements=default_measurements(),
    )


def form_values_from_state(state: WorkspaceState) -> dict[str, str]:
    form_values = platform_main.default_form_values()
    form_values.update(
        {
            "ga_weeks": str(state.ga_weeks),
            "ga_days": str(state.ga_days),
            "field_strength": state.field_strength,
            "motion_artifact": state.motion_artifact,
            "reference_profile": state.reference_profile,
        }
    )

    for definition in MEASUREMENT_DEFINITIONS:
        value = state.measurements.get(definition.parameter_id)
        form_values[definition.parameter_id] = "" if value is None else str(value)

    return form_values
@app.post("/generate-report", response_class=HTMLResponse)
async def generate_report(
    ga: str = Form(...), 
    left_ventricle: str = Form(...), 
    right_ventricle: str = Form(...)
):
    prompt = f"""
    You are an expert pediatric neuroradiologist writing a hospital-grade report.
    Analyze these specific fetal brain MRI findings:
    - Gestational Age (GA): {ga} weeks
    - Left Ventricular Atrial Diameter: {left_ventricle} mm
    - Right Ventricular Atrial Diameter: {right_ventricle} mm
    
    Structure your response into these exact 3 concise sections:
    1. **METRICS SUMMARY**: List the inputs cleanly.
    2. **CLINICAL FINDINGS**: State if the measurements indicate standard thresholds. For context, use these rules:
       - If either ventricle is >= 15.0mm, state it is Severe Ventriculomegaly (Giorgione 2022).
       - If between 10.0mm and 14.9mm, state it is Mild-to-Moderate Ventriculomegaly (Pagani 2014).
       - If the difference between left and right exceeds 2.0mm, note Asymmetric Atrial Diameters (Barzilay 2017).
    3. **RECOMMENDED NEXT CLINICAL STEPS**: Write 2-3 bullet points outlining the immediate diagnostic next steps (e.g., follow-up fetal echocardiography, neurosurgical consultation, or standard tracking) to maximize safety.
    
    Format the entire output using clean HTML tags like <p>, <ul>, <li>, and <strong>. 
    Do NOT include markdown triple backticks (```html). Keep it clean, professional, and dense.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        ai_text = response.text
    except Exception as e:
        ai_text = f"<p class='text-red-600 font-semibold'>Error connecting to AI brain: {str(e)}. Please check your API key configuration.</p>"
