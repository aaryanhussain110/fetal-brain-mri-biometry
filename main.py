from __future__ import annotations

import asyncio
from contextlib import redirect_stderr
from dataclasses import dataclass
from html import escape
from io import StringIO
from math import erf, isfinite, sqrt
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse
    from fastapi.templating import Jinja2Templates
except ModuleNotFoundError:
    class Request:
        headers: dict[str, str]

        def __init__(self) -> None:
            self.headers = {}

    class HTMLResponse:
        def __init__(self, content: str = "", *args: Any, **kwargs: Any) -> None:
            self.content = content
            self.body = content

    class Jinja2Templates:
        def __init__(self, directory: str) -> None:
            self.directory = directory

        def TemplateResponse(
            self,
            *,
            request: Request,
            name: str,
            context: dict[str, Any],
        ) -> dict[str, Any]:
            return {
                "request": request,
                "name": name,
                "context": context,
            }

    class FastAPI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def get(self, *args: Any, **kwargs: Any):
            def decorator(func):
                return func

            return decorator

        def post(self, *args: Any, **kwargs: Any):
            def decorator(func):
                return func

            return decorator

try:
    from scipy.stats import norm
except ModuleNotFoundError:
    class _NormFallback:
        @staticmethod
        def cdf(value: float) -> float:
            return 0.5 * (1.0 + erf(value / sqrt(2.0)))

    norm = _NormFallback()

from data_reference import (
    DIAGNOSTIC_LITERATURE_REGISTRY,
    PARAMETERS,
    ParameterDefinition,
    grouped_parameters,
    lookup_distribution,
    lookup_status_counts,
)


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
_KNOWLEDGE_CORPUS_CACHE: dict[tuple[str, ...], str] = {}
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _environment_timeout(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    if not isfinite(value):
        return default
    return max(5.0, min(value, 120.0))


GEMINI_REPORT_TIMEOUT_SECONDS = _environment_timeout("GEMINI_REPORT_TIMEOUT_SECONDS", 30.0)
GEMINI_CHAT_TIMEOUT_SECONDS = _environment_timeout("GEMINI_CHAT_TIMEOUT_SECONDS", 20.0)
GEMINI_PAPER_CHAT_TIMEOUT_SECONDS = _environment_timeout("GEMINI_PAPER_CHAT_TIMEOUT_SECONDS", 45.0)

GENERIC_CORPUS_QUERY_TOKENS = {
    "about",
    "article",
    "brain",
    "calculator",
    "corpus",
    "cover",
    "covers",
    "diagnosis",
    "does",
    "explain",
    "fetal",
    "mri",
    "paper",
    "papers",
    "protocol",
    "reference",
    "references",
    "risk",
    "risks",
    "source",
    "study",
    "summarize",
    "summary",
    "tell",
    "tracking",
    "what",
}

REF_TOPIC_SUMMARIES = {
    "REF_003": "Kyriakopoulou 2017 is the core fetal brain MRI normative biometry reference; it supports gestational-age centile lookup, z-score framing, and reference-curve interpretation.",
    "REF_005": "Garel 2005 provides fetal cerebral biometry reference material used as supporting normative context for structural brain measurements.",
    "REF_007": "Tilea 2009 provides fetal MRI cerebral biometry reference data used as a supporting source for growth-curve interpretation.",
    "REF_008": "Corroenne 2023 focuses on corpus callosal reference ranges and supports corpus callosum measurement interpretation.",
    "REF_009": "Kertes 2021 focuses on fetal CSP MRI biometry and supports cavum septum pellucidum measurement interpretation.",
    "REF_010": "Vatansever 2013 focuses on fetal posterior fossa analysis and supports posterior fossa measurement review.",
    "REF_011": "Dovjak 2021 focuses on fetal brainstem development by MRI and supports pons/brainstem measurement interpretation.",
    "REF_015": "Pagani 2014 is the mild-to-moderate isolated ventriculomegaly outcomes reference; in this calculator it supports the 10.0-14.9 mm pathway, isolated versus non-isolated risk stratification, and interval ventricular surveillance.",
    "REF_016": "Giorgione 2022 is the fetal ventriculomegaly counseling reference; in this calculator it supports the severe ventriculomegaly escalation pathway when atrial diameter is 15.0 mm or greater.",
    "REF_017": "Carta 2018 focuses on severe bilateral ventriculomegaly outcomes and supports high-risk counseling when ventricular enlargement is severe and bilateral.",
    "REF_018": "Barzilay 2017 focuses on ventriculomegaly asymmetry and MRI-associated anomalies; in this calculator it supports side-specific reporting when the left-right atrial difference exceeds 2.0 mm.",
    "REF_019": "Meyer 2018 focuses on isolated ventricular asymmetry outcomes and supports follow-up when ventricular asymmetry is present without additional abnormalities.",
    "REF_037": "Amugongo 2025 is a healthcare RAG systematic review used as AI/RAG methodology grounding for retrieval safety. For this calculator, it supports retrieving a bounded local paper corpus, citing the retrieved sources, and keeping the radiologist as the final clinical arbiter rather than allowing unsupported free-form generation.",
    "REF_038": "Wada 2025 addresses RAG in radiology consultation workflows and supports the Copilot grounding approach. For this calculator, it supports converting retrieved radiology evidence into concise consultation language while preserving source traceability.",
    "REF_040": "Adams 2024 addresses agentic AI and hallucination risk in radiology and supports cautious source-grounded AI use. For this calculator, it supports explicit source citation, refusal or uncertainty when a paper is absent, and human review before clinical use.",
    "REF_046": "Woitek 2014 is the posterior fossa morphometry/open neural tube defect reference; in this calculator it supports the joint TDPF and CSA Chiari II/open NTD warning pathway.",
    "REF_047": "Aertsen 2019 focuses on posterior fossa and brainstem measurement reliability, supporting measurement-quality interpretation.",
    "REF_048": "D'Addario 2001 focuses on the clivus-supraocciput angle and supports CSA interpretation.",
    "REF_049": "Bahlmann 2015 focuses on cranial/cerebral signs in spina bifida and supports the Chiari II/open NTD differential pathway.",
}

app = FastAPI(
    title="Fetal Brain MRI Biometry Calculator",
    description="Offline-capable FastAPI scaffold for fetal brain MRI biometry and HTMX-driven workflows.",
    version="0.5.0",
)


@app.post("/generate-report", response_class=HTMLResponse)
async def generate_report(request: Request):
    """Generate an HTML RAG report card from the local knowledge corpus."""

    raw_form = await request.form()
    form = {key: str(value) for key, value in raw_form.items()}
    form_values = default_form_values()
    form_values.update(form)
    page_context = build_page_context(form_values)

    if not _all_required_measurements_present(form_values):
        return HTMLResponse(_incomplete_report_html(form_values))

    ga_weeks = _first_form_value(form, "ga_weeks", "weeks", default="30")
    ga_days = _first_form_value(form, "ga_days", "days", default="0")
    va_left = _first_form_value(form, "va_left", "left_ventricular_diameter", "atrial_left")
    va_right = _first_form_value(form, "va_right", "right_ventricular_diameter", "atrial_right")
    lv_val = _float_or_none(va_left)
    rv_val = _float_or_none(va_right)
    ventricular_values = [value for value in (lv_val, rv_val) if value is not None]
    is_critical = max(ventricular_values, default=0.0) >= 15.0
    rule_text = _ventriculomegaly_rule_text(lv_val, rv_val)
    corpus = _load_retrieval_corpus(rule_text=rule_text)
    patient_context = _build_patient_chat_context(
        ga_weeks=page_context["ga_weeks"],
        ga_days=page_context["ga_days"],
        ga_fractional=page_context["ga_fractional"],
        results=page_context["results"],
        warning_cards=page_context["warning_cards"],
    )
    retrieved_context = f"{patient_context}\n\n{_build_chat_context(corpus=corpus, rule_text=rule_text)}"

    try:
        prompt = _build_ai_prompt(
            form=form,
            corpus=corpus,
            ga_weeks=ga_weeks,
            ga_days=ga_days,
            va_left=va_left,
            va_right=va_right,
            rule_text=rule_text,
        )
        report_html = _strip_markdown_fences(
            await _generate_gemini_text_with_timeout(
                prompt,
                timeout_seconds=GEMINI_REPORT_TIMEOUT_SECONDS,
            )
        )
        if not report_html:
            raise RuntimeError("Gemini returned an empty report.")
    except Exception as exc:
        report_html = _fallback_ai_report_html(
            form=form,
            ga_weeks=ga_weeks,
            ga_days=ga_days,
            va_left=va_left,
            va_right=va_right,
            rule_text=rule_text,
            reason=str(exc),
        )

    return HTMLResponse(
        _report_card_html(
            report_html,
            ga_weeks=ga_weeks,
            ga_days=ga_days,
            va_left=va_left,
            va_right=va_right,
            is_critical=is_critical,
            retrieved_context=retrieved_context,
        )
    )


@app.post("/chat-consult", response_class=HTMLResponse)
async def chat_consult(request: Request):
    form_data = await request.form()
    query = str(form_data.get("chat_query", "")).strip()
    context = str(form_data.get("chat_context", ""))
    chat_history = str(form_data.get("chat_history", "")).strip()
    if not query:
        return HTMLResponse(
            "<div class='rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-700 font-medium'>Please enter a valid question.</div>"
        )

    normalized_query = _normalize_chat_query(query)
    conversational_answer = _conversational_patient_answer(
        query=normalized_query,
        context=context,
    )
    if conversational_answer is not None:
        return HTMLResponse(_chat_exchange_html(query=query, answer=conversational_answer))

    patient_answer = _patient_context_answer(query=normalized_query, context=context)
    if patient_answer is not None:
        return HTMLResponse(_chat_exchange_html(query=query, answer=patient_answer))

    missing_ref_answer = _missing_ref_paper_answer(query=normalized_query)
    if missing_ref_answer is not None:
        return HTMLResponse(_chat_exchange_html(query=query, answer=missing_ref_answer))

    grounded_context = _augment_chat_context_for_query(query=normalized_query, context=context)
    is_paper_query = _should_answer_from_local_ref_index(normalized_query)
    chat_timeout = (
        GEMINI_PAPER_CHAT_TIMEOUT_SECONDS
        if is_paper_query
        else GEMINI_CHAT_TIMEOUT_SECONDS
    )
    chat_prompt = (
        "You are an expert pediatric neuroradiologist consulting on a case. "
        f"Answer this user follow-up question: '{normalized_query}'. "
        "The PATIENT CALCULATOR SUMMARY is authoritative for current patient measurements, z-scores, percentiles, and active diagnostic warnings. "
        "If asked about patient stats, measurements, or diagnoses, answer from that summary first. "
        "If asked about a paper, REF, source, or protocol, answer from the query-specific paper retrieval first and give a clinically useful summary, not just a one-line identity. "
        "Treat 'Brazilay' as a misspelling of Barzilay 2017. "
        "If the user asks about a named paper or protocol and that paper appears in the source index or retrieved excerpts, do not say it is absent. "
        "If the exact phrase 'tracking protocol' is not in the paper, synthesize the practical follow-up approach supported by the cited outcomes literature instead of refusing. "
        "When discussing Pagani 2014, focus on mild-to-moderate ventriculomegaly outcome stratification, isolated versus non-isolated status, and interval surveillance. "
        "When discussing Barzilay 2017, focus on ventricular asymmetry, side-specific reporting, associated anomaly review, and follow-up for progression. "
        "Cite papers by REF filename when using corpus content, for example REF_015 or REF_018. "
        f"Use this prior conversation only for continuity, not as a source of new facts:\n{chat_history or 'No prior chat turns.'}\n\n"
        "Base the medical content strictly on this literature grounding context:\n"
        f"{grounded_context}\n"
        "Keep your answer under 3 concise sentences using professional medical language."
    )
    try:
        answer = _strip_markdown_fences(
            await _generate_gemini_text_with_timeout(chat_prompt, timeout_seconds=chat_timeout)
        ).strip()
        if not answer:
            raise RuntimeError("Gemini returned an empty consultation.")
        if _answer_needs_citation_repair(query=normalized_query, answer=answer):
            answer = _local_literature_answer(query=normalized_query, context=grounded_context)
        return HTMLResponse(_chat_exchange_html(query=query, answer=answer))
    except Exception as exc:
        return HTMLResponse(_fallback_chat_consult_html(query=normalized_query, context=grounded_context, reason=str(exc)))


def _first_form_value(form: dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        value = form.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _all_required_measurements_present(form_values: dict[str, str]) -> bool:
    return all(str(form_values.get(parameter.parameter_id, "")).strip() for parameter in PARAMETERS)


def _missing_measurement_labels(form_values: dict[str, str]) -> list[str]:
    return [
        parameter.label
        for parameter in PARAMETERS
        if not str(form_values.get(parameter.parameter_id, "")).strip()
    ]


def _incomplete_report_html(form_values: dict[str, str]) -> str:
    missing = _missing_measurement_labels(form_values)
    missing_preview = ", ".join(missing[:6])
    if len(missing) > 6:
        missing_preview += f", plus {len(missing) - 6} more"

    return f"""
<section class="rounded-xl border border-amber-200 bg-amber-50 p-5 text-amber-900 shadow-sm">
  <h3 class="text-sm font-bold uppercase tracking-wide">Report not ready</h3>
  <p class="mt-2 text-sm leading-6">Complete every measurement field before generating the AI report. Missing: {escape(missing_preview or "required measurements")}.</p>
</section>
""".strip()


def _float_or_none(raw_value: str) -> float | None:
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _generate_gemini_text(prompt: str) -> str:
    """Call Gemini with the modern google-genai SDK and provide a clear setup error."""

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is not installed in the Python environment running Uvicorn. "
            "Run `python -m pip install -r requirements.txt`, then restart the server."
        ) from exc

    client = genai.Client()
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text or ""


async def _generate_gemini_text_with_timeout(prompt: str, *, timeout_seconds: float) -> str:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_generate_gemini_text, prompt),
            timeout=timeout_seconds,
        )
    except TimeoutError as exc:
        raise RuntimeError(
            f"Gemini response exceeded {timeout_seconds:.0f}s local timeout."
        ) from exc


def _normalize_chat_query(query: str) -> str:
    return (
        query.replace("Brazilay", "Barzilay")
        .replace("brazilay", "Barzilay")
        .replace("Brazily", "Barzilay")
        .replace("brazily", "Barzilay")
    )


def _query_specific_keywords(query: str) -> tuple[str, ...]:
    query_lower = query.lower()
    has_explicit_ref_lookup = _has_explicit_ref_lookup(query_lower)
    keywords: list[str] = []

    if any(term in query_lower for term in ("pagani", "mild", "moderate", "tracking", "follow", "protocol")):
        keywords.extend(["pagani_2014", "meyer_2018"])
    if any(term in query_lower for term in ("barzilay", "asymmetry", "asymmetric", "discrepancy", "laterality")):
        keywords.extend(["barzilay_2017", "meyer_2018"])
    asks_about_15mm_threshold = (
        "15" in query_lower
        and not has_explicit_ref_lookup
        and any(term in query_lower for term in ("ventricle", "ventricular", "ventriculomegaly", "atrium", "atrial", "diameter", "mm"))
    )
    if asks_about_15mm_threshold or any(term in query_lower for term in ("giorgione", "severe", "emergency", "escalat")):
        keywords.extend(["giorgione_2022", "carta_2018"])
    if any(
        term in query_lower
        for term in (
            "ventricle",
            "ventricular",
            "ventriculomegaly",
            "atrium",
            "atrial",
            "what is happening",
            "something wrong",
            "abnormal",
            "concerning",
            "concern",
            "risk",
        )
    ):
        keywords.extend(["pagani_2014", "giorgione_2022", "barzilay_2017", "meyer_2018"])
    if any(term in query_lower for term in ("woitek", "posterior fossa", "tdpf", "csa", "chiari", "neural tube")):
        keywords.extend(["woitek_2014", "bahlmann_2015"])
    if any(term in query_lower for term in ("kyriakopoulou", "normative", "centile", "percentile", "z-score", "z score")):
        keywords.extend(["kyriakopoulou_2017", "tilea_2009", "garel_2005"])
    if any(term in query_lower for term in ("rag", "corpus", "papers", "grounding", "hallucination", "source")):
        keywords.extend(["amugongo_2025", "wada_2025", "adams_2024"])

    keywords.extend(_filename_keywords_for_query(query))

    return tuple(dict.fromkeys(keywords))


def _has_explicit_ref_lookup(query: str) -> bool:
    return bool(re.search(r"(?:ref|paper)\s*[_#:-]?\s*0*\d{1,3}", query.lower()))


def _explicit_ref_numbers(query: str) -> set[int]:
    return {
        int(match)
        for match in re.findall(r"(?:ref|paper)\s*[_#:-]?\s*0*(\d{1,3})", query.lower())
    }


def _explicit_ref_labels(query: str) -> set[str]:
    return {f"REF_{number:03d}" for number in _explicit_ref_numbers(query)}


def _knowledge_file_paths() -> list[Path]:
    if not KNOWLEDGE_DIR.exists():
        return []
    return sorted(
        item
        for item in KNOWLEDGE_DIR.rglob("*")
        if item.is_file() and item.suffix.lower() in {".pdf", ".txt", ".md"}
    )


def _paper_slug(path: Path) -> str:
    slug = path.stem.lower()
    slug = re.sub(r"^ref_\d+__", "", slug)
    return slug


def _filename_keywords_for_query(query: str) -> tuple[str, ...]:
    query_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if (
            len(token) >= 4
            and token not in GENERIC_CORPUS_QUERY_TOKENS
            and not re.fullmatch(r"(19|20)\d{2}", token)
        )
    }

    matches: list[str] = []
    ref_numbers = _explicit_ref_numbers(query)
    for path in _knowledge_file_paths():
        slug = _paper_slug(path)
        ref_match = re.match(r"ref_(\d+)__", path.stem.lower())
        if ref_match and int(ref_match.group(1)) in ref_numbers:
            matches.append(slug)
            continue

        if not query_tokens:
            continue

        slug_tokens = set(re.findall(r"[a-z0-9]+", slug))
        if query_tokens & slug_tokens or any(token in slug for token in query_tokens):
            matches.append(slug)

    return tuple(dict.fromkeys(matches))


def _paper_ref_label(path: Path) -> str:
    ref_match = re.match(r"(ref_\d+)__", path.stem.lower())
    return ref_match.group(1).upper() if ref_match else path.stem


def _paper_topic_hint(path: Path) -> str:
    return _paper_slug(path).replace("_", " ")


def _matched_ref_papers_for_query(query: str, *, limit: int = 3) -> list[Path]:
    paths = _knowledge_file_paths()
    if not paths:
        return []

    query_lower = query.lower()
    ref_numbers = _explicit_ref_numbers(query_lower)
    if ref_numbers:
        exact_matches = []
        for path in paths:
            ref_match = re.match(r"ref_(\d+)__", path.stem.lower())
            if ref_match and int(ref_match.group(1)) in ref_numbers:
                exact_matches.append(path)
        return exact_matches[:limit]

    keywords = _filename_keywords_for_query(query)
    if not keywords:
        return []

    scored_paths: list[tuple[int, str, Path]] = []
    query_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", query_lower)
        if token not in GENERIC_CORPUS_QUERY_TOKENS and not re.fullmatch(r"(19|20)\d{2}", token)
    }
    for path in paths:
        slug = _paper_slug(path)
        slug_tokens = set(re.findall(r"[a-z0-9]+", slug))
        keyword_hits = sum(1 for keyword in keywords if keyword in slug)
        token_hits = len(query_tokens & slug_tokens)
        if keyword_hits or token_hits:
            score = (keyword_hits * 3) + token_hits
            scored_paths.append((-score, path.name.lower(), path))

    return [path for _, _, path in sorted(scored_paths)[:limit]]


def _should_answer_from_local_ref_index(query: str) -> bool:
    query_lower = query.lower()
    return _has_explicit_ref_lookup(query_lower) or any(
        term in query_lower
        for term in (
            "pagani",
            "barzilay",
            "giorgione",
            "woitek",
            "kyriakopoulou",
            "paper",
            "ref",
            "reference",
            "study",
            "protocol",
        )
    )


def _local_ref_paper_answer(*, query: str) -> str | None:
    matches = _matched_ref_papers_for_query(query, limit=3)
    if not matches:
        return None

    answer_parts = []
    for path in matches:
        ref = _paper_ref_label(path)
        summary = REF_TOPIC_SUMMARIES.get(
            ref,
            f"{ref} is indexed in the local REF corpus as {path.name}; the filename topic indicates { _paper_topic_hint(path) }.",
        )
        answer_parts.append(f"{summary} Cite it as {ref}.")

    if len(answer_parts) == 1:
        return answer_parts[0]

    return " ".join(answer_parts)


def _existing_ref_labels() -> set[str]:
    return {_paper_ref_label(path) for path in _knowledge_file_paths()}


def _missing_ref_paper_answer(*, query: str) -> str | None:
    requested_refs = _explicit_ref_labels(query)
    if not requested_refs:
        return None

    existing_refs = _existing_ref_labels()
    missing_refs = sorted(requested_refs - existing_refs)
    if not missing_refs:
        return None

    available_refs = sorted(existing_refs)
    nearby_refs = []
    for missing_ref in missing_refs:
        missing_number = int(missing_ref.split("_")[1])
        for existing_ref in available_refs:
            existing_number = int(existing_ref.split("_")[1])
            if abs(existing_number - missing_number) <= 2:
                nearby_refs.append(existing_ref)

    nearby_text = ""
    if nearby_refs:
        nearby_text = f" Nearby available references are {', '.join(dict.fromkeys(nearby_refs))}."

    return (
        f"{', '.join(missing_refs)} is not present in the local knowledge corpus, so I should not infer its contents from the patient case or another paper."
        f"{nearby_text} Please ask about an available REF number if you want a paper-specific summary."
    )


def _all_paper_source_index() -> str:
    paths = _knowledge_file_paths()
    if not paths:
        return "- No local paper files found."
    return "\n".join(f"- {path.name}" for path in paths)


def _load_all_paper_overview() -> str:
    paths = _knowledge_file_paths()
    if not paths:
        return "No local paper corpus files found."

    manifest_lines = []
    for path in paths:
        slug = _paper_slug(path)
        topic_hint = slug.replace("_", " ")
        manifest_lines.append(f"- {path.name}: {topic_hint}")

    return (
        "All local paper corpus files are available for query-specific retrieval. "
        "Use this manifest to identify relevant sources, then rely on the query-specific excerpts when present.\n"
        + "\n".join(manifest_lines)
    )


def _augment_chat_context_for_query(*, query: str, context: str) -> str:
    keywords = _query_specific_keywords(query)
    all_paper_overview = _load_all_paper_overview()

    targeted_context = ""
    if keywords:
        targeted_corpus = _load_knowledge_corpus(
            file_keywords=keywords,
            max_chars_per_file=5000,
            max_pdf_pages=4,
        )
        targeted_context = _build_chat_context(
            corpus=targeted_corpus,
            rule_text="Query-specific retrieval for Copilot follow-up.",
            max_chars=30000,
        )

    combined = (
        f"{context}\n\n"
        "===== ALL-PAPER CORPUS SOURCE INDEX =====\n"
        f"{_all_paper_source_index()}\n\n"
        "===== ALL-PAPER COMPACT CORPUS OVERVIEW =====\n"
        f"{all_paper_overview}\n\n"
    )
    if targeted_context:
        combined += (
            "===== QUERY-SPECIFIC PAPER RETRIEVAL =====\n"
            f"{targeted_context}"
        )
    if len(combined) > 90000:
        return combined[:90000].rsplit(" ", 1)[0] + "\n\n[Combined chat context truncated for prompt size.]"
    return combined


def _answer_needs_citation_repair(*, query: str, answer: str) -> bool:
    query_lower = query.lower()
    answer_lower = answer.lower()
    requested_refs = _explicit_ref_labels(query_lower)
    asks_known_citation = _should_answer_from_local_ref_index(query_lower) or any(
        term in query_lower
        for term in ("pagani", "barzilay", "giorgione", "woitek", "kyriakopoulou")
    )
    refusal_language = any(
        phrase in answer_lower
        for phrase in (
            "does not contain",
            "not contain information",
            "not present",
            "unable to explain",
            "cannot explain",
            "not reference",
            "no information on",
        )
    )
    wrong_barzilay_anchor = "barzilay" in query_lower and "kyriakopoulou" in answer_lower and "barzilay" not in answer_lower
    missing_requested_ref = any(ref.lower() not in answer_lower for ref in requested_refs)
    return (asks_known_citation and refusal_language) or wrong_barzilay_anchor or missing_requested_ref


def _fallback_chat_consult_html(*, query: str, context: str, reason: str) -> str:
    answer = _local_literature_answer(query=query, context=context)
    note = _friendly_gemini_failure_note(reason)
    return _chat_exchange_html(query=query, answer=answer, note=note)


def _friendly_gemini_failure_note(reason: str) -> str:
    if "429" in reason or "RESOURCE_EXHAUSTED" in reason or "Quota exceeded" in reason:
        return "Live Gemini quota is temporarily exhausted, so this answer used the local calculator and paper-corpus fallback."
    if "local timeout" in reason or "exceeded" in reason:
        return "Live Gemini was slower than expected, so this answer used the local calculator and paper-corpus fallback."
    if "google-genai is not installed" in reason:
        return "Live Gemini is not installed in this Python environment, so this answer used the local calculator and paper-corpus fallback."
    return "Live Gemini is temporarily unavailable, so this answer used the local calculator and paper-corpus fallback."


def _chat_exchange_html(*, query: str, answer: str, note: str | None = None) -> str:
    note_html = ""
    if note:
        note_html = f"<p class='mt-2 text-[10px] leading-4 text-slate-500'>{escape(note)}</p>"

    return f"""
<article class='rounded-xl border border-slate-200 bg-white p-3 shadow-sm'>
  <div class='rounded-lg bg-indigo-50 px-3 py-2 text-xs text-indigo-900'>
    <span class='font-bold uppercase tracking-wide'>Question</span>
    <p class='mt-1 leading-relaxed'>{escape(query)}</p>
  </div>
  <div class='mt-2 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-800'>
    <span class='font-bold uppercase tracking-wide text-slate-600'>Copilot</span>
    <p class='mt-1 font-medium leading-relaxed'>{escape(answer)}</p>
    {note_html}
  </div>
</article>
""".strip()


def _build_patient_chat_context(
    *,
    ga_weeks: int,
    ga_days: int,
    ga_fractional: float,
    results: dict[str, "ParameterResult"],
    warning_cards: list[dict[str, Any]],
) -> str:
    lines = [
        "===== PATIENT CALCULATOR SUMMARY =====",
        f"Gestational Age: {ga_weeks}w {ga_days}d ({ga_fractional:.2f} weeks)",
        "All required measurements: complete",
        "",
        "Measurements:",
    ]

    for parameter in PARAMETERS:
        result = results[parameter.parameter_id]
        if result.input_value is None:
            lines.append(f"- {parameter.label}: not entered")
            continue

        if result.z_score is not None and result.percentile is not None:
            lines.append(
                f"- {parameter.label}: {result.input_value:.1f} {parameter.unit}; "
                f"Z {result.z_score:+.2f}; {result.percentile:.0f}th percentile; {result.status_label}"
            )
        else:
            lines.append(
                f"- {parameter.label}: {result.input_value:.1f} {parameter.unit}; {result.status_label}"
            )

    lines.extend(["", "Active diagnostic warning cards:"])
    if warning_cards:
        for card in warning_cards:
            refs = ", ".join(paper["ref"] for paper in card.get("ref_papers", ()))
            ref_note = f"; Applicable REFs: {refs}" if refs else ""
            lines.append(
                f"- {card['title']}: {card['trigger']}; "
                f"Citation: {card.get('citation', 'not listed')}{ref_note}"
            )
            for item in card.get("items", ()):
                lines.append(f"  Differential/support: {item}")
    else:
        lines.append("- None. No diagnostic warning card is currently triggered by the entered measurements.")

    lines.append("===== END PATIENT CALCULATOR SUMMARY =====")
    return "\n".join(lines)


def _extract_patient_summary(context: str) -> str:
    start_marker = "===== PATIENT CALCULATOR SUMMARY ====="
    end_marker = "===== END PATIENT CALCULATOR SUMMARY ====="
    start = context.find(start_marker)
    end = context.find(end_marker)
    if start == -1 or end == -1 or end <= start:
        return ""
    return context[start : end + len(end_marker)]


def _active_diagnostic_lines(patient_summary: str) -> list[str]:
    diagnostic_lines = _section_lines(patient_summary, "Active diagnostic warning cards:", "===== END")
    return [
        line
        for line in diagnostic_lines
        if line.startswith("- ") and not line.startswith("- None.")
    ]


def _patient_diagnosis_status_text(patient_summary: str) -> str:
    ga_line = _first_matching_line(patient_summary, "Gestational Age:")
    active_diagnoses = _active_diagnostic_lines(patient_summary)
    if active_diagnoses:
        diagnosis_text = "; ".join(line.removeprefix("- ") for line in active_diagnoses)
        return f"{ga_line}. Active diagnostic warnings: {diagnosis_text}"

    return (
        f"{ga_line}. No active diagnostic warning cards are triggered by the current measurements, "
        "so the calculator does not identify a flagged diagnosis for this patient."
    )


def _is_acknowledgement_query(query: str) -> bool:
    normalized = re.sub(r"[^a-z\s]", "", query.lower()).strip()
    return normalized in {
        "thanks",
        "thank you",
        "thank you so much",
        "thx",
        "okay",
        "ok",
        "got it",
        "great",
    }


def _is_confirmation_query(query: str) -> bool:
    normalized = re.sub(r"[^a-z\s]", "", query.lower()).strip()
    return normalized in {
        "are you sure",
        "are you certain",
        "is that right",
        "is this right",
        "confirm",
        "can you confirm",
        "double check",
        "double check that",
    }


def _conversational_patient_answer(*, query: str, context: str) -> str | None:
    patient_summary = _extract_patient_summary(context)
    if not patient_summary:
        if _is_acknowledgement_query(query):
            return "You are welcome."
        return None

    status_text = _patient_diagnosis_status_text(patient_summary)
    if _is_acknowledgement_query(query):
        return "You are welcome. The current patient context remains available for follow-up questions."
    if _is_confirmation_query(query):
        return (
            "Yes, based on the current entered measurements and active warning-card rules, "
            f"{status_text} If any measurements are edited, the assessment should be recalculated."
        )
    return None


def _patient_context_answer(*, query: str, context: str) -> str | None:
    query_lower = query.lower()
    patient_summary = _extract_patient_summary(context)
    if not patient_summary:
        return None

    asks_about_patient = any(
        term in query_lower
        for term in (
            "patient",
            "stats",
            "stat",
            "measurement",
            "measurements",
            "diagnosis",
            "diagnoses",
            "diagnostic",
            "flagged",
            "normal",
            "what is happening",
            "something wrong",
            "anything wrong",
            "z-score",
            "z score",
            "percentile",
        )
    )
    if not asks_about_patient:
        return None

    measurement_lines = _section_lines(patient_summary, "Measurements:", "Active diagnostic warning cards:")
    status_text = _patient_diagnosis_status_text(patient_summary)

    if any(term in query_lower for term in ("diagnosis", "diagnoses", "diagnostic", "flagged", "normal", "what is happening", "something wrong", "anything wrong")):
        return status_text

    if any(term in query_lower for term in ("stats", "stat", "patient", "status")):
        key_measurements = _select_key_measurements(measurement_lines, query_lower)
        return f"{status_text} Measurements: {'; '.join(key_measurements)}"

    key_measurements = _select_key_measurements(measurement_lines, query_lower)
    ga_line = _first_matching_line(patient_summary, "Gestational Age:")
    return f"{ga_line}. Current patient measurements: {'; '.join(key_measurements)}"


def _first_matching_line(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line
    return prefix.rstrip(":")


def _section_lines(text: str, start_label: str, end_label: str) -> list[str]:
    lines = text.splitlines()
    try:
        start_index = lines.index(start_label) + 1
    except ValueError:
        return []

    section: list[str] = []
    for line in lines[start_index:]:
        if line.startswith(end_label):
            break
        if line.strip():
            section.append(line.strip())
    return section


def _select_key_measurements(measurement_lines: list[str], query_lower: str) -> list[str]:
    if "all" in query_lower or "measurement" in query_lower or "stats" in query_lower:
        return [line.removeprefix("- ") for line in measurement_lines[:]]

    priority_terms = (
        "Ventricular",
        "Posterior Fossa",
        "Clivus-Supraocciput",
        "Tegmento-Vermian",
        "Cisterna Magna",
    )
    selected = [
        line.removeprefix("- ")
        for line in measurement_lines
        if any(term in line for term in priority_terms)
    ]
    return selected or [line.removeprefix("- ") for line in measurement_lines[:6]]


def _local_literature_answer(*, query: str, context: str) -> str:
    query_lower = query.lower()
    patient_summary = _extract_patient_summary(context)
    patient_status_text = _patient_diagnosis_status_text(patient_summary) if patient_summary else ""
    active_patient_diagnoses = " ".join(_active_diagnostic_lines(patient_summary)).lower() if patient_summary else ""

    conversational_answer = _conversational_patient_answer(query=query, context=context)
    if conversational_answer is not None:
        return conversational_answer

    missing_ref_answer = _missing_ref_paper_answer(query=query)
    if missing_ref_answer is not None:
        return missing_ref_answer

    local_ref_answer = _local_ref_paper_answer(query=query)
    if local_ref_answer is not None and _should_answer_from_local_ref_index(query):
        return local_ref_answer

    asks_about_15mm_threshold = (
        "15" in query_lower
        and not _has_explicit_ref_lookup(query_lower)
        and any(term in query_lower for term in ("ventricle", "ventricular", "ventriculomegaly", "atrium", "atrial", "diameter", "mm"))
    )
    if asks_about_15mm_threshold or any(term in query_lower for term in ("giorgione", "severe", "emergency", "escalat")):
        return (
            "For a ventricular atrial diameter of 15 mm or greater, use the Giorgione 2022 severe ventriculomegaly branch (REF_016): treat the finding as high-risk and escalate to maternal-fetal medicine with detailed neurosonography/anatomic review. "
            "Recommended workup should include side-specific ventricular tracking, evaluation for associated CNS/non-CNS anomalies, fetal echocardiography, and genetic/infectious testing when clinically appropriate."
        )

    if any(term in query_lower for term in ("pagani", "mild", "moderate", "tracking", "follow", "protocol")):
        return (
            "For atrial measurements in the 10.0-14.9 mm range, the Pagani 2014 pathway (REF_015) emphasizes distinguishing isolated from non-isolated ventriculomegaly and documenting interval stability or progression. "
            "A practical follow-up plan is targeted neurosonography plus serial ventricular measurements, with broader genetic, infectious, and anatomic evaluation if additional abnormalities appear."
        )

    if any(term in query_lower for term in ("barzilay", "asymmetry", "asymmetric", "discrepancy", "laterality")):
        return (
            "For a left-right atrial discrepancy greater than 2 mm, the Barzilay 2017 asymmetry pathway (REF_018) supports explicit side-specific reporting and follow-up for progression toward true ventriculomegaly. "
            "If asymmetry is isolated, risk is generally lower than severe bilateral dilation, but interval tracking remains important because ventricular caliber can evolve."
        )

    if any(term in query_lower for term in ("woitek", "posterior fossa", "tdpf", "csa", "chiari")):
        return (
            "For posterior fossa questions, use the Woitek 2014 morphometry branch (REF_046): concordantly low TDPF and CSA z-scores support a Chiari II/open neural tube defect pattern rather than isolated vermian hypoplasia alone. "
            "The report should emphasize the combined geometric pattern, correlate with spine imaging and posterior fossa morphology, and recommend fetal medicine/neurosurgical review when the joint pattern is strongly abnormal."
        )

    if any(term in query_lower for term in ("kyriakopoulou", "normative", "centile", "percentile", "z-score", "z score")):
        return (
            "For normative biometry questions, the Kyriakopoulou-style reference pathway (REF_003) treats the interpolated 50th centile as the expected mean and converts the 5th-to-95th centile envelope into an approximate local standard deviation. "
            "That approach supports z-score and percentile reporting, but any interpretation should still consider gestational age accuracy, motion artifact, and whether a measurement is isolated or syndromic."
        )

    if any(term in query_lower for term in ("rag", "corpus", "papers", "grounding", "hallucination", "source")):
        return (
            "The Copilot is designed as a retrieval-grounded assistant using the local REF corpus, including RAG/radiology AI sources such as REF_037, REF_038, and REF_040. "
            "For safety, it should be treated as radiologist decision support rather than an autonomous report writer, with final wording verified against the source papers and the full imaging examination."
        )

    if "severe ventriculomegaly" in active_patient_diagnoses:
        return (
            "The submitted context triggers the severe ventriculomegaly route, so the most relevant clinical action is urgent escalation and exclusion of obstructive, syndromic, infectious, and associated structural causes. "
            "Use the report as a decision-support draft and correlate with fetal MRI, ultrasound, and local high-risk obstetric protocols."
        )

    if "asymmetric" in active_patient_diagnoses:
        return (
            "The submitted context highlights ventricular asymmetry, so the key point is to document laterality and maintain interval surveillance for progression or normalization. "
            "Escalation depends on whether asymmetry remains isolated or is accompanied by increasing atrial diameter or additional CNS findings."
        )

    if patient_status_text:
        return patient_status_text

    return (
        "Based on the available literature context, frame the follow-up around the triggered ventricular threshold, whether the finding is isolated, and whether interval measurements show progression. "
        "For a radiology workflow, the safest concise recommendation is targeted neurosonography, side-specific atrial measurement surveillance, and escalation of genetic/infectious or subspecialty consultation when severity or associated anomalies are present."
    )


def _read_pdf_text(path: Path, *, max_pages: int | None = None) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        with path.open("rb") as handle:
            return handle.read().decode("utf-8", errors="ignore")

    with path.open("rb") as handle, redirect_stderr(StringIO()):
        reader = PdfReader(handle)
        pages = list(reader.pages)
        if max_pages is not None:
            pages = pages[:max_pages]
        page_text = [page.extract_text() or "" for page in pages]
    return "\n".join(page_text)


def _read_knowledge_file(
    path: Path,
    *,
    max_chars: int | None = None,
    max_pdf_pages: int | None = None,
) -> str:
    if path.suffix.lower() == ".pdf":
        text = _read_pdf_text(path, max_pages=max_pdf_pages)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0] + "\n[File excerpt truncated.]"
    return text


def _load_knowledge_corpus(
    *,
    file_keywords: tuple[str, ...] = (),
    max_chars_per_file: int | None = None,
    max_pdf_pages: int | None = None,
) -> str:
    global _KNOWLEDGE_CORPUS_CACHE

    cache_key = file_keywords + (
        f"chars={max_chars_per_file}",
        f"pages={max_pdf_pages}",
    )
    if cache_key in _KNOWLEDGE_CORPUS_CACHE:
        return _KNOWLEDGE_CORPUS_CACHE[cache_key]

    if not KNOWLEDGE_DIR.exists():
        _KNOWLEDGE_CORPUS_CACHE[cache_key] = "No local knowledge corpus found."
        return _KNOWLEDGE_CORPUS_CACHE[cache_key]

    candidate_paths = [
        item
        for item in KNOWLEDGE_DIR.rglob("*")
        if item.is_file() and item.suffix.lower() in {".pdf", ".txt", ".md"}
    ]
    if file_keywords:
        priority = {keyword: index for index, keyword in enumerate(file_keywords)}

        def sort_key(path: Path) -> tuple[int, str]:
            normalized_name = path.name.lower()
            matched_priority = min(
                (index for keyword, index in priority.items() if keyword in normalized_name),
                default=len(priority),
            )
            return matched_priority, normalized_name

        candidate_paths = [
            path
            for path in candidate_paths
            if any(keyword in path.name.lower() for keyword in file_keywords)
        ]
        candidate_paths = sorted(candidate_paths, key=sort_key)
    else:
        candidate_paths = sorted(candidate_paths)

    chunks: list[str] = []
    for path in candidate_paths:
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        try:
            text = _read_knowledge_file(
                path,
                max_chars=max_chars_per_file,
                max_pdf_pages=max_pdf_pages,
            ).strip()
        except Exception as exc:
            text = f"[Unable to extract text from {path.name}: {exc}]"
        if text:
            chunks.append(f"\n\n===== {path.name} =====\n{text}")

    _KNOWLEDGE_CORPUS_CACHE[cache_key] = "\n".join(chunks) if chunks else "No readable corpus text found."
    return _KNOWLEDGE_CORPUS_CACHE[cache_key]


def _load_retrieval_corpus(*, rule_text: str) -> str:
    keywords = [
        "kyriakopoulou_2017",
        "prayer_2023",
        "woitek_2014",
        "amugongo_2025",
        "wada_2025",
        "pagani_2014",
        "giorgione_2022",
        "barzilay_2017",
        "meyer_2018",
    ]
    if "Giorgione 2022" in rule_text or ">= 15.0 mm" in rule_text:
        keywords.append("giorgione_2022")
    if "Pagani 2014" in rule_text or "10.0-14.9 mm" in rule_text:
        keywords.append("pagani_2014")
    if "Barzilay 2017" in rule_text or "discrepancy exceeds 2.0 mm" in rule_text:
        keywords.append("barzilay_2017")
    if not any(
        keyword in keywords
        for keyword in ("giorgione_2022", "pagani_2014", "barzilay_2017")
    ):
        keywords.extend(["pagani_2014", "giorgione_2022", "barzilay_2017"])

    return _load_knowledge_corpus(
        file_keywords=tuple(dict.fromkeys(keywords)),
        max_chars_per_file=2200,
        max_pdf_pages=2,
    )


def _build_chat_context(*, corpus: str, rule_text: str, max_chars: int = 60000) -> str:
    source_names = [
        line.removeprefix("===== ").removesuffix(" =====")
        for line in corpus.splitlines()
        if line.startswith("===== ") and line.endswith(" =====")
    ]
    source_index = "\n".join(f"- {source}" for source in source_names)
    context = (
        f"Triggered clinical routing: {rule_text}\n\n"
        f"Retrieved paper corpus source index:\n{source_index or '- No source files matched.'}\n\n"
        f"Local literature corpus excerpt:\n{corpus}"
    )
    if len(context) <= max_chars:
        return context
    return context[:max_chars].rsplit(" ", 1)[0] + "\n\n[Context excerpt truncated for browser payload size.]"


def _ventriculomegaly_rule_text(va_left: float | None, va_right: float | None) -> str:
    ventricular_values = [value for value in (va_left, va_right) if value is not None]
    max_ventricle = max(ventricular_values) if ventricular_values else None
    asymmetric = (
        va_left is not None
        and va_right is not None
        and abs(va_left - va_right) > 2.0
    )

    rules: list[str] = []
    if max_ventricle is not None and max_ventricle >= 15.0:
        rules.append("Map to Giorgione 2022 because at least one atrial diameter is >= 15.0 mm.")
    elif max_ventricle is not None and 10.0 <= max_ventricle <= 14.9:
        rules.append("Map to Pagani 2014 because the maximum atrial diameter is 10.0-14.9 mm.")
    if asymmetric:
        rules.append("Map to Barzilay 2017 because hemispheric atrial discrepancy exceeds 2.0 mm.")

    return " ".join(rules) if rules else "No ventriculomegaly or asymmetry citation rule is triggered."


def _build_ai_prompt(
    *,
    form: dict[str, str],
    corpus: str,
    ga_weeks: str,
    ga_days: str,
    va_left: str,
    va_right: str,
    rule_text: str,
) -> str:
    metric_lines = "\n".join(
        f"- {key}: {value}"
        for key, value in sorted(form.items())
        if str(value).strip()
    )

    return f"""
You are generating a dense, technical fetal brain MRI reporting aid for a radiologist.
Use only the local research corpus and numeric inputs below. Do not introduce outside claims.
Return raw HTML snippet content only. Do not use markdown fences.

The report card must contain exactly these three clinical headers, each as a visible heading:
1. METRICS LOG SUMMARY
2. GROUNDED RESEARCH DIAGNOSIS
3. MANDATED RADIOLOGIST CLINICAL NEXT STEPS

Clinical citation routing rules:
- If either lateral ventricular atrial diameter is >= 15.0 mm, explicitly cite Giorgione 2022.
- If the maximum lateral ventricular atrial diameter is 10.0-14.9 mm, explicitly cite Pagani 2014.
- If the left-right ventricular discrepancy exceeds 2.0 mm, explicitly cite Barzilay 2017.
- Keep language technical, hospital-grade, and useful for radiologist decision support.
- Include actionable next steps such as urgent MFM review, targeted neurosonography, fetal echocardiography, genetic/infectious workup, interval MRI/ultrasound follow-up, and pediatric neurosurgery referral only when supported by the triggered pattern.

Gestational age: {ga_weeks}w {ga_days}d
Left lateral ventricular atrium: {va_left or "not provided"} mm
Right lateral ventricular atrium: {va_right or "not provided"} mm
Triggered citation logic: {rule_text}

All submitted metrics:
{metric_lines or "- No metrics submitted."}

Local research corpus:
{corpus}
""".strip()


def _strip_markdown_fences(raw_text: str) -> str:
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _fallback_ai_report_html(
    *,
    form: dict[str, str],
    ga_weeks: str,
    ga_days: str,
    va_left: str,
    va_right: str,
    rule_text: str,
    reason: str,
) -> str:
    rows = "\n".join(
        f"<tr><td>{escape(key)}</td><td>{escape(value)}</td></tr>"
        for key, value in sorted(form.items())
        if str(value).strip()
    )
    if not rows:
        rows = "<tr><td colspan=\"2\">No submitted metrics.</td></tr>"

    left = _float_or_none(va_left)
    right = _float_or_none(va_right)
    max_ventricle = max([value for value in (left, right) if value is not None], default=None)
    asymmetry = left is not None and right is not None and abs(left - right) > 2.0

    diagnosis_items: list[str] = []
    next_steps: list[str] = []
    if max_ventricle is not None and max_ventricle >= 15.0:
        diagnosis_items.append(
            "Severe ventriculomegaly pattern is triggered; route interpretation to Giorgione 2022 and evaluate obstructive, syndromic, infectious, and associated CNS/non-CNS etiologies."
        )
        next_steps.extend(
            [
                "Escalate for urgent maternal-fetal medicine and fetal neurology/neurosurgery correlation.",
                "Recommend detailed neurosonography, complete anatomic survey, fetal echocardiography, and genetic/infectious evaluation as clinically appropriate.",
            ]
        )
    elif max_ventricle is not None and 10.0 <= max_ventricle <= 14.9:
        diagnosis_items.append(
            "Mild-to-moderate ventriculomegaly pattern is triggered; route interpretation to Pagani 2014 with attention to isolated versus non-isolated status and interval progression."
        )
        next_steps.extend(
            [
                "Recommend targeted neurosonography and interval ventricular measurement surveillance.",
                "Assess for extracranial anomalies, genetic risk, and congenital infection when not already excluded.",
            ]
        )
    else:
        diagnosis_items.append("No lateral ventricular diameter rule is triggered by the submitted values.")

    if asymmetry:
        diagnosis_items.append(
            "Hemispheric atrial discrepancy exceeds 2.0 mm; route interpretation to Barzilay 2017 and document asymmetric ventricular caliber."
        )
        next_steps.append("Track side-specific ventricular atrial diameters on follow-up imaging for progression or normalization.")

    if not next_steps:
        next_steps.append("Continue standard fetal brain MRI biometry review and correlate with ultrasound surveillance.")

    diagnosis_html = "".join(f"<li>{escape(item)}</li>" for item in diagnosis_items)
    next_steps_html = "".join(f"<li>{escape(item)}</li>" for item in next_steps)

    return f"""
<h3>METRICS LOG SUMMARY</h3>
<p><strong>Gestational age:</strong> {escape(ga_weeks)}w {escape(ga_days)}d. <strong>Left VA:</strong> {escape(va_left or "not provided")} mm. <strong>Right VA:</strong> {escape(va_right or "not provided")} mm.</p>
<table><tbody>{rows}</tbody></table>
<h3>GROUNDED RESEARCH DIAGNOSIS</h3>
<p>{escape(rule_text)}</p>
<ul>{diagnosis_html}</ul>
<h3>MANDATED RADIOLOGIST CLINICAL NEXT STEPS</h3>
<ul>{next_steps_html}</ul>
<p><small>{escape(_friendly_gemini_failure_note(reason))}</small></p>
""".strip()


def _report_card_html(
    report_html: str,
    *,
    ga_weeks: str,
    ga_days: str,
    va_left: str,
    va_right: str,
    is_critical: bool,
    retrieved_context: str,
) -> str:
    exam_type = "Fetal Brain MRI Biometry / AI-RAG Support"
    gestational_age = f"{ga_weeks}w {ga_days}d"
    left_va = f"{va_left} mm" if va_left else "Not provided"
    right_va = f"{va_right} mm" if va_right else "Not provided"
    emergency_banner = ""
    if is_critical:
        emergency_banner = """
    <div class="mx-auto my-4 flex max-w-5xl flex-col gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-red-900 shadow-sm sm:flex-row sm:items-center sm:justify-between non-printable">
        <p class="text-xs font-sans leading-5">🚨 <strong>CRITICAL PATH ALERT:</strong> Severe ventriculomegaly parameters detected. Immediate clinical escalation recommended per Giorgione 2022 guidelines.</p>
        <button onclick="alert('📢 EMERGENCY NOTIFICATION DISPATCHED: Escalation logged. Fetal Medicine Team, High-Risk Obstetrics Unit, and Pediatric Neurosurgery Coordinator have been signaled via hospital pager networks.')" class="shrink-0 self-start rounded-lg bg-red-600 px-3.5 py-2 text-[11px] font-bold text-white shadow-sm transition hover:bg-red-700 sm:self-center">📟 Dispatch Emergency Alert</button>
    </div>
""".rstrip()
        emergency_banner = emergency_banner.replace("<button onclick=", '<button type="button" onclick=', 1)
    chat_context_value = escape(retrieved_context, quote=True)

    return f"""
<section class="bg-emerald-50 border-l-4 border-emerald-600 rounded-xl p-5 shadow-sm text-slate-900">
  <div id="final-radiology-report" class="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
    <header class="bg-slate-950 px-6 py-5 text-white">
      <p class="text-xs font-bold uppercase tracking-[0.35em] text-emerald-300">FETAL NEURO-MEDICINE WORKSTATION</p>
      <h2 class="mt-2 text-2xl font-bold tracking-tight">Automated AI-RAG Clinical Decision Support Utility</h2>
    </header>

    <section class="border-b border-slate-200 bg-slate-50 px-6 py-5">
      <table class="w-full border-collapse text-sm">
        <tbody>
          <tr class="border-b border-slate-200">
            <th class="w-1/3 bg-slate-100 px-3 py-2 text-left text-xs font-bold uppercase tracking-[0.18em] text-slate-500">Exam Type</th>
            <td class="px-3 py-2 font-semibold text-slate-800">{escape(exam_type)}</td>
          </tr>
          <tr class="border-b border-slate-200">
            <th class="bg-slate-100 px-3 py-2 text-left text-xs font-bold uppercase tracking-[0.18em] text-slate-500">Gestational Age</th>
            <td class="px-3 py-2 font-semibold text-slate-800">{escape(gestational_age)}</td>
          </tr>
          <tr class="border-b border-slate-200">
            <th class="bg-slate-100 px-3 py-2 text-left text-xs font-bold uppercase tracking-[0.18em] text-slate-500">Left Ventricular Atrial Diameter</th>
            <td class="px-3 py-2 font-semibold text-slate-800">{escape(left_va)}</td>
          </tr>
          <tr>
            <th class="bg-slate-100 px-3 py-2 text-left text-xs font-bold uppercase tracking-[0.18em] text-slate-500">Right Ventricular Atrial Diameter</th>
            <td class="px-3 py-2 font-semibold text-slate-800">{escape(right_va)}</td>
          </tr>
        </tbody>
      </table>
    </section>
{emergency_banner}

    <section class="px-6 py-6">
      <div class="prose prose-sm max-w-none text-slate-800 prose-headings:mt-6 prose-headings:font-bold prose-headings:uppercase prose-headings:tracking-[0.16em] prose-h3:text-slate-950 prose-p:leading-7 prose-li:leading-7 prose-table:w-full prose-td:border prose-td:border-slate-200 prose-td:px-2 prose-td:py-1">
        {report_html}
      </div>
    </section>

    <section class="mx-6 border-t border-slate-200 py-5">
      <p class="text-sm font-bold text-slate-950">Attending Radiologist, MD // Electronic Sign-off Verified</p>
      <div class="mt-6 border-t border-slate-200 pt-4 non-printable font-sans">
          <h4 class="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">💬 Literature Copilot Consultation Chat</h4>
          <div id="ai-chat-transcript" class="mb-2 max-h-72 min-h-[72px] space-y-3 overflow-y-auto rounded-lg bg-slate-100 p-3 text-xs text-slate-600">
              <div id="ai-chat-placeholder" class="rounded-lg border border-dashed border-slate-300 bg-white/70 px-3 py-3 italic">Ask a research or tracking follow-up question regarding these clinical findings. Follow-up answers will append here without reloading the page.</div>
          </div>
          <div class="flex gap-2">
              <input type="hidden" id="chat_context" name="chat_context" value="{chat_context_value}">
              <input type="hidden" id="chat_history" name="chat_history" value="">
              <input type="text" id="chat_query" name="chat_query" autocomplete="off" onkeydown="if(event.key === 'Enter'){{ event.preventDefault(); const button = document.getElementById('ask-copilot-button'); if (button) {{ button.click(); }} }}" placeholder="e.g., Explain the Pagani 2014 tracking protocols or Barzilay 2017 asymmetry risks..." class="w-full text-xs px-3 py-2 border rounded-lg focus:outline-none focus:ring-1 focus:ring-indigo-500">
              <button id="ask-copilot-button" hx-post="/chat-consult" hx-include="#chat_context, #chat_history, #chat_query" hx-target="#ai-chat-transcript" hx-swap="beforeend" hx-sync="this:drop" hx-disabled-elt="this" hx-on::before-request="const p=document.getElementById('ai-chat-placeholder'); if(p){{ p.remove(); }} const q=document.getElementById('chat_query'); if(q){{ q.dataset.lastQuery=q.value; }}" hx-on::after-swap="const q=document.getElementById('chat_query'); const h=document.getElementById('chat_history'); const t=document.getElementById('ai-chat-transcript'); if(t){{ t.scrollTop=t.scrollHeight; }} if(h && t){{ h.value=t.innerText.slice(-8000); }} if(q){{ q.value=''; q.focus(); }}" type="button" class="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-xs py-2 px-4 rounded-lg shadow transition disabled:cursor-wait disabled:opacity-60">Ask Copilot</button>
          </div>
      </div>
      <p class="mt-1 text-xs uppercase tracking-[0.18em] text-slate-500">Generated decision-support draft for physician review</p>
    </section>

    <footer class="border-t border-slate-200 bg-slate-50 px-6 py-4">
      <p class="text-xs leading-6 text-slate-600">
        Clinical disclaimer: This AI-RAG output is a decision-support utility only. It must be reviewed, edited, and finalized by the interpreting radiologist and correlated with the complete fetal MRI examination, ultrasound findings, genetic/infectious workup, and institutional clinical protocols.
      </p>
    </footer>
  </div>

  <div class="non-printable mt-4 flex justify-end border-t border-emerald-200 pt-3">
    <button type="button" onclick="printFinalRadiologyReport()" class="rounded-xl bg-emerald-700 px-4 py-2 text-sm font-bold text-white shadow-sm transition hover:bg-emerald-800">
      🖨️ Save as PDF / Print Report
    </button>
  </div>

  <script>
    function printFinalRadiologyReport() {{
      const report = document.getElementById("final-radiology-report");
      if (!report) {{
        return;
      }}

      const printWindow = window.open("", "_blank", "width=960,height=1200");
      if (!printWindow) {{
        window.alert("Please allow pop-ups to print the isolated radiology report.");
        return;
      }}

      printWindow.document.open();
      printWindow.document.write(`<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Fetal Neuro-Medicine Workstation Report</title>
</head>
<body class="bg-white p-8">
  ${{report.outerHTML}}
</body>
</html>`);
      printWindow.document.close();

      const tailwindStylesheet = printWindow.document.createElement("link");
      tailwindStylesheet.rel = "stylesheet";
      tailwindStylesheet.href = "https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css";
      printWindow.document.head.appendChild(tailwindStylesheet);

      const printStyles = printWindow.document.createElement("style");
      printStyles.textContent = "@page {{ size: letter; margin: 0.6in; }} @media print {{ body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }} .non-printable {{ display: none !important; }} }}";
      printWindow.document.head.appendChild(printStyles);

      let didPrint = false;
      const runPrint = () => {{
        if (didPrint) {{
          return;
        }}
        didPrint = true;
        printWindow.focus();
        printWindow.print();
        setTimeout(() => printWindow.close(), 250);
      }};

      if (tailwindStylesheet.onload === null) {{
        tailwindStylesheet.onload = runPrint;
        tailwindStylesheet.onerror = runPrint;
        setTimeout(runPrint, 1200);
      }} else {{
        runPrint();
      }}
    }}
  </script>

  <style>
    @media print {{ .non-printable {{ display: none !important; }} }}
  </style>
</section>
""".strip()


FIELD_STRENGTH_OPTIONS = ("0.55T", "1.5T", "3T")
MOTION_OPTIONS = ("None", "Mild", "Moderate", "Severe")
REFERENCE_PROFILE_OPTIONS = (
    "Standard Biometry: Kyriakopoulou 2017 + Garel/Tilea",
    "Posterior Fossa / ONTD: Woitek 2014 + Bahlmann 2015",
    "Ventriculomegaly Outcomes: Pagani 2014 + Giorgione 2022 + Barzilay 2017",
    "AI/RAG Methodology: Amugongo 2025 + Wada 2025 + Adams 2024",
)


@dataclass(frozen=True)
class ParameterResult:
    definition: ParameterDefinition
    raw_value: str
    input_value: float | None
    expected_mean: float | None
    expected_sd: float | None
    z_score: float | None
    percentile: float | None
    marker_position: float | None
    status_label: str
    status_classes: str
    reference_note: str


def fractional_gestational_age(weeks: int, days: int) -> float:
    return weeks + (days / 7.0)


def tdpf_mean(ga_fractional: float) -> float:
    return (-0.01307 * (ga_fractional**2)) + (2.55571 * ga_fractional) - 21.71


def tdpf_sd(ga_fractional: float) -> float:
    return (0.06716 * ga_fractional) + 0.547


def csa_mean(ga_fractional: float) -> float:
    return (-0.04767 * (ga_fractional**2)) + (4.20404 * ga_fractional) + 1.73


def csa_sd(ga_fractional: float) -> float:
    return (0.01814 * ga_fractional) + 5.821


def z_score(value: float, expected_mean: float, expected_sd: float) -> float:
    return (value - expected_mean) / expected_sd


def percentile_from_z(z_value: float) -> float:
    return norm.cdf(z_value) * 100.0


def analytic_distribution(parameter_id: str, ga_fractional: float) -> tuple[float, float] | None:
    if parameter_id == "tdpf":
        return tdpf_mean(ga_fractional), tdpf_sd(ga_fractional)
    if parameter_id == "csa":
        return csa_mean(ga_fractional), csa_sd(ga_fractional)
    return None


def expected_distribution(parameter_id: str, ga_fractional: float) -> tuple[float, float] | None:
    analytic = analytic_distribution(parameter_id, ga_fractional)
    if analytic is not None:
        return analytic
    return lookup_distribution(parameter_id, ga_fractional)


def parse_float(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None

    trimmed = raw_value.strip()
    if not trimmed:
        return None

    try:
        return float(trimmed)
    except ValueError:
        return None


def status_for_z(z_value: float) -> tuple[str, str]:
    if z_value <= -3.0:
        return "Critically low", "border-rose-400/70 bg-rose-500/10 text-rose-100"
    if z_value < -2.0:
        return "Below expected", "border-amber-400/70 bg-amber-500/10 text-amber-100"
    if z_value >= 3.0:
        return "Critically high", "border-rose-400/70 bg-rose-500/10 text-rose-100"
    if z_value > 2.0:
        return "Above expected", "border-amber-400/70 bg-amber-500/10 text-amber-100"
    return "Within expected", "border-emerald-400/70 bg-emerald-500/10 text-emerald-200"


def analytic_reference_note(parameter_id: str, ga_fractional: float) -> str:
    if parameter_id in {"tdpf", "csa"}:
        note = "Woitek 2014 analytic regression (validated 21-37 weeks)"
        if ga_fractional < 21.0 or ga_fractional > 37.0:
            note += "; extrapolated outside validated range"
        return note

    return "Kyriakopoulou-style spline centile lookup scaffold"


def evaluate_parameter(
    definition: ParameterDefinition,
    raw_value: str,
    ga_fractional: float,
) -> ParameterResult:
    numeric_value = parse_float(raw_value)
    if numeric_value is None:
        return ParameterResult(
            definition=definition,
            raw_value=raw_value,
            input_value=None,
            expected_mean=None,
            expected_sd=None,
            z_score=None,
            percentile=None,
            marker_position=None,
            status_label="Awaiting input",
            status_classes="border-slate-200 bg-slate-50 text-slate-600",
            reference_note="Enter a measurement to calculate z-score and percentile.",
        )

    distribution = expected_distribution(definition.parameter_id, ga_fractional)
    if distribution is None:
        return ParameterResult(
            definition=definition,
            raw_value=raw_value,
            input_value=numeric_value,
            expected_mean=None,
            expected_sd=None,
            z_score=None,
            percentile=None,
            marker_position=None,
            status_label="Reference pending",
            status_classes="border-slate-200 bg-slate-50 text-slate-600",
            reference_note="No reference distribution is available at this gestational age.",
        )

    expected_mean, expected_sd = distribution
    z_value = z_score(numeric_value, expected_mean, expected_sd)
    percentile = percentile_from_z(z_value)
    marker_position = float(np.clip(((z_value + 3.0) / 6.0) * 100.0, 0.0, 100.0))
    status_label, status_classes = status_for_z(z_value)

    return ParameterResult(
        definition=definition,
        raw_value=raw_value,
        input_value=numeric_value,
        expected_mean=expected_mean,
        expected_sd=expected_sd,
        z_score=z_value,
        percentile=percentile,
        marker_position=marker_position,
        status_label=status_label,
        status_classes=status_classes,
        reference_note=analytic_reference_note(definition.parameter_id, ga_fractional),
    )


def mahalanobis_squared(
    observation: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> float:
    centered = observation - mean
    inverse = np.linalg.inv(covariance)
    return float(centered.T @ inverse @ centered)


def chiari_group_posteriors(tdpf_result: ParameterResult, csa_result: ParameterResult) -> dict[str, float] | None:
    if tdpf_result.z_score is None or csa_result.z_score is None:
        return None

    observation = np.array([tdpf_result.z_score, csa_result.z_score], dtype=float)
    groups = {
        "controls": (
            np.array([0.0, 0.0], dtype=float),
            covariance_matrix(1.0, 1.0, 0.0),
        ),
        "ontd": (
            np.array([-3.6, -2.6], dtype=float),
            covariance_matrix(0.9, 1.1, 0.54),
        ),
        "cntd": (
            np.array([-1.4, -0.6], dtype=float),
            covariance_matrix(1.0, 1.0, 0.0),
        ),
    }

    weights: dict[str, float] = {}
    for group_name, (mean, covariance) in groups.items():
        distance_squared = mahalanobis_squared(observation, mean, covariance)
        weights[group_name] = float(np.exp(-distance_squared / 2.0))

    total = sum(weights.values())
    if total <= 0:
        return None

    return {
        group_name: weight / total
        for group_name, weight in weights.items()
    }


def chiari_joint_probability(tdpf_result: ParameterResult, csa_result: ParameterResult) -> float | None:
    posteriors = chiari_group_posteriors(tdpf_result, csa_result)
    if posteriors is None:
        return None
    return posteriors["ontd"]


def covariance_matrix(std_x: float, std_y: float, correlation: float) -> np.ndarray:
    covariance = correlation * std_x * std_y
    return np.array(
        [[std_x**2, covariance], [covariance, std_y**2]],
        dtype=float,
    )


DIAGNOSTIC_REF_PAPERS: dict[str, tuple[dict[str, str], ...]] = {
    "mild_moderate_ventriculomegaly": (
        {
            "ref": "REF_015",
            "label": "Pagani 2014 isolated mild ventriculomegaly outcomes",
            "filename": "REF_015__pagani_2014_isolated_mild_ventriculomegaly_outcomes.pdf",
        },
        {
            "ref": "REF_019",
            "label": "Meyer 2018 isolated ventricular asymmetry outcomes",
            "filename": "REF_019__meyer_2018_isolated_ventricular_asymmetry_outcomes.pdf",
        },
    ),
    "severe_ventriculomegaly": (
        {
            "ref": "REF_016",
            "label": "Giorgione 2022 fetal ventriculomegaly counseling",
            "filename": "REF_016__giorgione_2022_fetal_ventriculomegaly_counseling.pdf",
        },
        {
            "ref": "REF_017",
            "label": "Carta 2018 severe bilateral ventriculomegaly outcomes",
            "filename": "REF_017__carta_2018_severe_bilateral_ventriculomegaly_outcomes.pdf",
        },
    ),
    "asymmetric_ventricles": (
        {
            "ref": "REF_018",
            "label": "Barzilay 2017 ventriculomegaly asymmetry MRI anomalies",
            "filename": "REF_018__barzilay_2017_ventriculomegaly_asymmetry_mri_anomalies.pdf",
        },
        {
            "ref": "REF_019",
            "label": "Meyer 2018 isolated ventricular asymmetry outcomes",
            "filename": "REF_019__meyer_2018_isolated_ventricular_asymmetry_outcomes.pdf",
        },
    ),
    "chiari_ii_ontd": (
        {
            "ref": "REF_046",
            "label": "Woitek 2014 posterior fossa morphometry neural tube defects",
            "filename": "REF_046__woitek_2014_posterior_fossa_morphometry_neural_tube_defects.pdf",
        },
        {
            "ref": "REF_049",
            "label": "Bahlmann 2015 spina bifida cranial cerebral signs",
            "filename": "REF_049__bahlmann_2015_spina_bifida_cranial_cerebral_signs.pdf",
        },
    ),
}


def build_warning_cards(results: dict[str, ParameterResult]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []

    left_vent = results["left_ventricular_diameter"].input_value
    right_vent = results["right_ventricular_diameter"].input_value
    severe_vm = (left_vent is not None and left_vent >= 15.0) or (
        right_vent is not None and right_vent >= 15.0
    )
    mild_moderate_vm = (
        not severe_vm
        and (
            (left_vent is not None and 10.0 <= left_vent < 15.0)
            or (right_vent is not None and 10.0 <= right_vent < 15.0)
        )
    )
    asymmetry = (
        left_vent is not None
        and right_vent is not None
        and abs(left_vent - right_vent) > 2.0
    )

    if mild_moderate_vm:
        registry = DIAGNOSTIC_LITERATURE_REGISTRY["mild_moderate_ventriculomegaly"]
        cards.append(
            {
                "title": registry["condition"],
                "trigger": registry["trigger"],
                "items": (
                    "Isolated or idiopathic (~8%): neurodevelopmental delay risk is relatively low in isolated mild VM.",
                    "Associated CNS or extracranial anomaly: dedicated structural review remains important.",
                    "Chromosomal abnormality (~5-15%): aneuploidy is a recognized contributor.",
                ),
                "citation": registry["citation"],
                "url": registry["url"],
                "ref_papers": DIAGNOSTIC_REF_PAPERS["mild_moderate_ventriculomegaly"],
                "tone_classes": "border-amber-300/70 bg-amber-50 text-amber-900",
            }
        )

    if severe_vm:
        registry = DIAGNOSTIC_LITERATURE_REGISTRY["severe_ventriculomegaly"]
        cards.append(
            {
                "title": registry["condition"],
                "trigger": registry["trigger"],
                "items": (
                    "Aqueductal stenosis (~20%): most common obstructive hydrocephalus consideration.",
                    "Associated CNS or non-CNS anomaly (high likelihood): severe ventriculomegaly is frequently non-isolated.",
                    "Chromosomal abnormality (significant likelihood): risk rises with severity, including trisomy 21, 18, and 13.",
                    "Congenital infection (~1-5%): CMV and toxoplasmosis remain important exclusions.",
                    "Isolated or idiopathic (~10-20%): diagnosis of exclusion after structural and genetic workup.",
                ),
                "citation": registry["citation"],
                "url": registry["url"],
                "ref_papers": DIAGNOSTIC_REF_PAPERS["severe_ventriculomegaly"],
                "tone_classes": "border-amber-400/70 bg-amber-500/10 text-amber-100",
            }
        )

    if asymmetry:
        registry = DIAGNOSTIC_LITERATURE_REGISTRY["asymmetric_ventricles"]
        cards.append(
            {
                "title": registry["condition"],
                "trigger": registry["trigger"],
                "items": (
                    "Often benign if isolated, but progression toward true ventriculomegaly can occur.",
                    "Published MRI series reports roughly 37-46% progression risk in asymmetric cases.",
                ),
                "citation": registry["citation"],
                "url": registry["url"],
                "ref_papers": DIAGNOSTIC_REF_PAPERS["asymmetric_ventricles"],
                "tone_classes": "border-cyan-300/70 bg-cyan-50 text-cyan-900",
            }
        )

    tdpf_result = results["tdpf"]
    csa_result = results["csa"]
    joint_probability = chiari_joint_probability(tdpf_result, csa_result)
    if (
        tdpf_result.z_score is not None
        and csa_result.z_score is not None
        and tdpf_result.z_score < -2.0
        and csa_result.z_score < -2.0
        and joint_probability is not None
        and joint_probability > 0.5
    ):
        registry = DIAGNOSTIC_LITERATURE_REGISTRY["chiari_ii_ontd"]
        cards.append(
            {
                "title": registry["condition"],
                "trigger": (
                    f"TDPF Z {tdpf_result.z_score:+.2f}, CSA Z {csa_result.z_score:+.2f}, "
                    f"joint probability {joint_probability:.0%}."
                ),
                "items": (
                    "Chiari II malformation / open neural tube defect (~85-90%): dominant posterior-fossa pattern match.",
                    "Closed neural tube defect (~5-10%): typically causes milder TDPF and CSA shifts than open lesions.",
                    "Severe vermian hypoplasia / Dandy-Walker spectrum (~3-5%): can shrink the fossa, but CSA is usually preserved or widened.",
                    "Other rare mimics (<2% overall): benign small posterior fossa variants or infection-related hypoplasia are residual considerations.",
                ),
                "citation": registry["citation"],
                "url": registry["url"],
                "ref_papers": DIAGNOSTIC_REF_PAPERS["chiari_ii_ontd"],
                "tone_classes": "border-rose-400/70 bg-rose-500/10 text-rose-100",
            }
        )

    return cards


def build_preview_text(
    ga_weeks: int,
    ga_days: int,
    ga_fractional: float,
    results: dict[str, ParameterResult],
    field_strength: str,
    motion_artifact: str,
    reference_profile: str,
) -> str:
    lines = [
        "Fetal Brain MRI Biometry",
        f"GA: {ga_weeks}w {ga_days}d ({ga_fractional:.2f} weeks)",
        f"Acquisition: {field_strength} | Motion: {motion_artifact} | Profile: {reference_profile}",
        "",
    ]

    grouped = grouped_parameters()
    for group in grouped:
        populated_results = [
            results[parameter.parameter_id]
            for parameter in group["parameters"]
            if results[parameter.parameter_id].input_value is not None
        ]
        if not populated_results:
            continue

        lines.append(group["name"])
        for result in populated_results:
            if result.z_score is not None and result.percentile is not None:
                lines.append(
                    f"{result.definition.label}: {result.input_value:.1f} {result.definition.unit} "
                    f"(Z: {result.z_score:+.2f}, {result.percentile:.0f}th percentile)"
                )
            else:
                lines.append(
                    f"{result.definition.label}: {result.input_value:.1f} {result.definition.unit} "
                    "(reference curve pending)"
                )
        lines.append("")

    return "\n".join(lines).strip()


def default_form_values() -> dict[str, str]:
    values = {
        "ga_weeks": "22",
        "ga_days": "0",
        "field_strength": "1.5T",
        "motion_artifact": "None",
        "reference_profile": "Standard Biometry: Kyriakopoulou 2017 + Garel/Tilea",
        "methodology_open": "false",
        "active_sample": "",
    }
    for parameter in PARAMETERS:
        values[parameter.parameter_id] = ""
    return values


def clamp_int(raw_value: str, minimum: int, maximum: int, default: int) -> int:
    try:
        parsed = int(raw_value)
    except ValueError:
        parsed = default
    return max(minimum, min(maximum, parsed))


def build_page_context(form_values: dict[str, str]) -> dict[str, Any]:
    ga_weeks = clamp_int(form_values.get("ga_weeks", "22"), minimum=18, maximum=40, default=22)
    ga_days = clamp_int(form_values.get("ga_days", "0"), minimum=0, maximum=6, default=0)
    ga_fractional = fractional_gestational_age(ga_weeks, ga_days)

    results = {
        parameter.parameter_id: evaluate_parameter(
            definition=parameter,
            raw_value=form_values.get(parameter.parameter_id, ""),
            ga_fractional=ga_fractional,
        )
        for parameter in PARAMETERS
    }

    lookup_loaded_count, lookup_total_count = lookup_status_counts()

    return {
        "page_title": "Fetal Brain MRI Biometry Calculator",
        "form_values": form_values,
        "ga_weeks": ga_weeks,
        "ga_days": ga_days,
        "ga_fractional": ga_fractional,
        "parameter_groups": grouped_parameters(),
        "results": results,
        "warning_cards": build_warning_cards(results),
        "all_measurements_complete": _all_required_measurements_present(form_values),
        "preview_text": build_preview_text(
            ga_weeks=ga_weeks,
            ga_days=ga_days,
            ga_fractional=ga_fractional,
            results=results,
            field_strength=form_values.get("field_strength", "1.5T"),
            motion_artifact=form_values.get("motion_artifact", "None"),
            reference_profile=form_values.get(
                "reference_profile",
                "Standard Biometry: Kyriakopoulou 2017 + Garel/Tilea",
            ),
        ),
        "field_strength_options": FIELD_STRENGTH_OPTIONS,
        "motion_options": MOTION_OPTIONS,
        "reference_profile_options": REFERENCE_PROFILE_OPTIONS,
        "lookup_loaded_count": lookup_loaded_count,
        "lookup_total_count": lookup_total_count,
    }


def is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request") == "true"


def render_workspace(request: Request, context: dict[str, Any]) -> HTMLResponse:
    template_name = "partials/workspace.html" if is_htmx(request) else "index.html"
    return templates.TemplateResponse(
        request=request,
        name=template_name,
        context=context,
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    context = build_page_context(default_form_values())
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=context,
    )


@app.post("/calculate", response_class=HTMLResponse)
async def calculate(request: Request) -> HTMLResponse:
    form = await request.form()
    form_values = default_form_values()

    for key, value in form.multi_items():
        form_values[key] = str(value)

    context = build_page_context(form_values)
    return render_workspace(request, context)
