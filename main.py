from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from urllib.parse import urlparse, quote

import os
import requests
import json
import re
from datetime import datetime
from typing import Optional, Literal, List, Any, Dict

from openai import OpenAI

# ------------------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------------------

app = FastAPI(
    title="FakeSite Detector Backend",
    version="0.2",
    description="Heuristik + optional LLM, mit zentralem Airtable-Cache pro URL."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP ok
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Konfiguration: Airtable & OpenAI
# ------------------------------------------------------------------------------

AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "Feedback")
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

LLM_SYSTEM_PROMPT = """
Du bist ein automatischer Klassifikator für potenziell betrügerische Webseiten.

Du bekommst einen AUSZUG aus dem Text einer Website (max. einige tausend Zeichen).
Deine Aufgabe ist, NUR anhand dieses Textes eine Risikoeinschätzung vorzunehmen UND typische Frames/Deutungsrahmen zu identifizieren.

Ziele:
- Erkenne, ob der Text harmlos, verdächtig oder sehr wahrscheinlich Betrug ist.
- Erkenne den Zweck der Seite (Shop, Bank/Login, Gewinnspiel, Crypto/Investment, Info, Sonstiges).
- Erkenne Ton und Muster:
  - Dringlichkeit / künstlicher Zeitdruck
  - Drohungen (z.B. Konto wird gesperrt)
  - Appell an Gier („schnell reich werden“, unrealistische Gewinne)
  - Forderung nach Geld, Zugangsdaten, persönlichen Daten oder Krypto-Transaktionen.
- Erkenne FRAMES: zentrale Begriffe oder Phrasen, die einen bestimmten Deutungsrahmen setzen.

Deine Ausgabe MUSS ein gültiges JSON-Objekt sein, OHNE zusätzlichen Text, OHNE Erklärungen, OHNE Markdown.

Schema:
{
  "overall_label": "safe" | "suspicious" | "scam",
  "overall_confidence": float (0.0 bis 1.0),
  "page_type": "shop" | "bank" | "crypto" | "giveaway" | "login" | "info" | "other",
  "tone": { "urgency": float, "threat": float, "greed_appeal": float },
  "asks_for": { "money": bool, "credentials": bool, "personal_data": bool, "crypto_transfer": bool },
  "risk_reasons": [ { "factor": "...", "confidence": float, "explanation": string } ],
  "frames": [ { "trigger": string, "frame_label": string, "explanation": string } ]
}
"""

# ------------------------------------------------------------------------------
# Airtable Helpers
# ------------------------------------------------------------------------------

def utc_iso_z() -> str:
    # Airtable-freundlich: "...Z" statt "+00:00"
    return datetime.utcnow().isoformat() + "Z"

def get_airtable_base_url() -> str:
    if not AIRTABLE_BASE_ID:
        raise RuntimeError("AIRTABLE_BASE_ID ist nicht gesetzt")
    if not AIRTABLE_TABLE_NAME:
        raise RuntimeError("AIRTABLE_TABLE_NAME ist nicht gesetzt")
    table_enc = quote(AIRTABLE_TABLE_NAME, safe="")
    return f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{table_enc}"

def airtable_headers() -> Dict[str, str]:
    if not AIRTABLE_TOKEN:
        raise RuntimeError("AIRTABLE_TOKEN ist nicht gesetzt")
    return {
        "Authorization": f"Bearer {AIRTABLE_TOKEN}",
        "Content-Type": "application/json",
    }

def airtable_escape_formula_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")

def url_to_key(url: str) -> str:
    """
    Canonical URL Key: scheme://host/path (ohne query/hash)
    """
    try:
        u = urlparse(url)
        scheme = u.scheme or "https"
        netloc = u.netloc
        path = u.path or "/"
        return f"{scheme}://{netloc}{path}"
    except Exception:
        return url

def airtable_find_summary_record(url_key: str) -> Optional[Dict[str, Any]]:
    base_url = get_airtable_base_url()
    headers = airtable_headers()

    key_esc = airtable_escape_formula_string(url_key)
    formula = f"AND({{Record Type}}='summary', {{URL Key}}='{key_esc}')"

    params = {"filterByFormula": formula, "maxRecords": 1}
    resp = requests.get(base_url, headers=headers, params=params, timeout=15)
    if not resp.ok:
        print("Airtable summary lookup error:", resp.status_code, resp.text)
        return None
    records = resp.json().get("records", [])
    return records[0] if records else None

def should_skip_llm_using_summary(summary_record: Optional[Dict[str, Any]]) -> bool:
    """
    Skip nur wenn:
    - Confirmed OK == True
    - False Positive Count == 0
    - Scam Count == 0
    - Last Score/Level/Reasons vorhanden
    """
    if not summary_record:
        return False
    f = summary_record.get("fields", {})
    confirmed = bool(f.get("Confirmed OK", False))
    fp = int(f.get("False Positive Count", 0) or 0)
    sc = int(f.get("Scam Count", 0) or 0)
    has_cached = ("Last Score" in f) and ("Last Level" in f) and ("Last Reasons" in f)
    return confirmed and fp == 0 and sc == 0 and has_cached

# ------------------------------------------------------------------------------
# Datenmodelle
# ------------------------------------------------------------------------------

class PageData(BaseModel):
    url: str
    text: str = ""
    hasPassword: bool = False
    hasCreditCard: bool = False

class FrameInfo(BaseModel):
    trigger: str
    frame_label: str
    explanation: str

class RiskResult(BaseModel):
    # ✅ neu für Extension-UX
    status: Literal["none", "pending", "done", "error"] = "done"
    fromCache: Optional[bool] = None
    url: Optional[str] = None

    score: int
    level: Literal["green", "yellow", "red"]
    reasons: List[str]

    llm_label: Optional[str] = None
    llm_confidence: Optional[float] = None
    llm_page_type: Optional[str] = None
    frames: Optional[List[FrameInfo]] = None

class Feedback(BaseModel):
    url: str
    backend_score: int
    backend_level: str
    backend_reasons: List[str]
    user_label: Literal["ok", "scam", "false_positive"]

# ------------------------------------------------------------------------------
# Summary <-> RiskResult
# ------------------------------------------------------------------------------

def result_from_summary(summary_record: Dict[str, Any], page_url: str, cached: bool) -> RiskResult:
    f = summary_record.get("fields", {})

    reasons_text = f.get("Last Reasons", "") or ""
    reasons = [r for r in reasons_text.split("\n") if r.strip()]

    frames_raw = f.get("Last Frames", "[]") or "[]"
    frames_list: Optional[List[FrameInfo]] = None
    try:
        parsed = json.loads(frames_raw)
        if isinstance(parsed, list) and parsed:
            frames_list = []
            for item in parsed[:10]:
                trig = (item.get("trigger") or "").strip()
                fl = (item.get("frame_label") or "").strip()
                ex = (item.get("explanation") or "").strip()
                if trig and fl and ex:
                    frames_list.append(FrameInfo(trigger=trig, frame_label=fl, explanation=ex))
    except Exception:
        frames_list = None

    llm_label = (f.get("Last LLM Label") or "").strip() or None
    llm_page_type = (f.get("Last LLM Page Type") or "").strip() or None
    llm_confidence = None
    try:
        v = f.get("Last LLM Confidence", None)
        llm_confidence = float(v) if v is not None else None
    except Exception:
        llm_confidence = None

    return RiskResult(
        status="done",
        fromCache=cached,
        url=page_url,
        score=int(f.get("Last Score", 0) or 0),
        level=(f.get("Last Level", "green") or "green"),
        reasons=reasons or ["(Cache) Keine Begründungen gespeichert."],
        llm_label=llm_label,
        llm_confidence=llm_confidence,
        llm_page_type=llm_page_type,
        frames=frames_list,
    )

def airtable_upsert_summary(url_key: str, page_url: str, result: RiskResult) -> None:
    base_url = get_airtable_base_url()
    headers = airtable_headers()

    existing = airtable_find_summary_record(url_key)

    frames_json = "[]"
    if result.frames:
        frames_json = json.dumps([f.model_dump() for f in result.frames], ensure_ascii=False)

    # Werte aus existing erhalten (Counts/Confirmed nicht überschreiben)
    prev_fields = (existing or {}).get("fields", {})
    confirmed_ok = bool(prev_fields.get("Confirmed OK", False))
    fp = int(prev_fields.get("False Positive Count", 0) or 0)
    sc = int(prev_fields.get("Scam Count", 0) or 0)

    fields = {
        "Record Type": "summary",
        "URL Key": url_key,

        "Confirmed OK": confirmed_ok,
        "False Positive Count": fp,
        "Scam Count": sc,

        "Last Score": int(result.score),
        "Last Level": result.level,
        "Last Reasons": "\n".join(result.reasons or []),
        "Last Frames": frames_json,

        "Last LLM Label": (result.llm_label or ""),
        "Last LLM Confidence": float(result.llm_confidence or 0.0),
        "Last LLM Page Type": (result.llm_page_type or ""),

        # ✅ Airtable-kompatibles Datum
        "Last Updated": utc_iso_z(),

        # optional
        "URL": page_url,
    }

    if existing:
        rec_id = existing["id"]
        patch_url = f"{base_url}/{rec_id}"
        resp = requests.patch(patch_url, headers=headers, json={"fields": fields}, timeout=15)
        if not resp.ok:
            print("Airtable summary patch error:", resp.status_code, resp.text)
    else:
        payload = {"records": [{"fields": fields}]}
        resp = requests.post(base_url, headers=headers, json=payload, timeout=15)
        if not resp.ok:
            print("Airtable summary create error:", resp.status_code, resp.text)

def airtable_update_summary_feedback(url_key: str, user_label: str) -> None:
    base_url = get_airtable_base_url()
    headers = airtable_headers()

    existing = airtable_find_summary_record(url_key)
    if not existing:
        return

    f = existing.get("fields", {})
    fp = int(f.get("False Positive Count", 0) or 0)
    sc = int(f.get("Scam Count", 0) or 0)
    confirmed = bool(f.get("Confirmed OK", False))

    if user_label == "ok":
        confirmed = True
    elif user_label == "false_positive":
        fp += 1
    elif user_label == "scam":
        sc += 1

    patch_fields = {
        "Confirmed OK": confirmed,
        "False Positive Count": fp,
        "Scam Count": sc,
        "Last Updated": utc_iso_z(),
    }

    patch_url = f"{base_url}/{existing['id']}"
    resp = requests.patch(patch_url, headers=headers, json={"fields": patch_fields}, timeout=15)
    if not resp.ok:
        print("Airtable summary feedback patch error:", resp.status_code, resp.text)

# ------------------------------------------------------------------------------
# LLM / Text
# ------------------------------------------------------------------------------

def truncate_text(text: str, max_chars: int = 4000) -> str:
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized if len(normalized) <= max_chars else normalized[:max_chars]

def analyze_with_llm(page_text: str) -> Optional[dict]:
    if not client or not OPENAI_API_KEY:
        return None
    snippet = truncate_text(page_text, max_chars=4000)
    if not snippet:
        return None

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": "Hier ist der Textauszug der Website:\n\n" + snippet},
            ],
        )
        raw_content = completion.choices[0].message.content or ""

        start = raw_content.find("{")
        end = raw_content.rfind("}")
        raw_json = raw_content[start:end + 1] if start != -1 and end != -1 else raw_content

        return json.loads(raw_json)
    except Exception as e:
        print("LLM-Analysefehler:", repr(e))
        return None

# ------------------------------------------------------------------------------
# Analysefunktion (Heuristik + optional LLM)
# ------------------------------------------------------------------------------

def analyze_page(data: PageData) -> RiskResult:
    url_score = 0
    content_score = 0
    form_score = 0
    reasons: List[str] = []

    parsed = urlparse(data.url)
    hostname = parsed.hostname or ""

    if len(hostname) > 25 or len(hostname) < 5:
        url_score += 10
        reasons.append("Ungewöhnlich lange oder sehr kurze Domain.")

    risky_tlds = [".top", ".shop", ".xyz", ".online", ".club"]
    if any(hostname.endswith(tld) for tld in risky_tlds):
        url_score += 10
        reasons.append("Top-Level-Domain ist häufiger bei Fake-Shops anzutreffen.")

    if re.search(r"[0-9-]{5,}", hostname):
        url_score += 10
        reasons.append("Viele Zahlen oder Bindestriche in der Domain.")

    if not data.url.startswith("https://"):
        url_score += 25
        reasons.append("Seite nutzt kein HTTPS (unsichere Verbindung).")

    text_lower = (data.text or "").lower()

    sales_patterns = [
        "nur heute", "limited offer", "jetzt zugreifen",
        "90% rabatt", "80% rabatt", "70% rabatt",
        "deal endet in", "nur für kurze zeit",
        "jetzt bestellen", "jetzt kaufen"
    ]
    if any(p in text_lower for p in sales_patterns):
        content_score += 20
        reasons.append("Aggressive Verkaufs- oder Druckformulierungen gefunden.")

    phishing_patterns = [
        "ihr konto wird gesperrt",
        "bestätigen sie ihre zugangsdaten",
        "verifizieren sie ihr konto",
        "melden sie sich erneut an",
        "aus sicherheitsgründen müssen sie"
    ]
    if any(p in text_lower for p in phishing_patterns):
        content_score += 30
        reasons.append("Typische Phishing-Formulierungen erkannt.")

    payment_patterns = [
        "nur vorkasse", "vorkasse", "sofortüberweisung",
        "zahlen sie jetzt", "sofort bezahlen",
        "überweisung innerhalb von 24 stunden"
    ]
    if any(p in text_lower for p in payment_patterns):
        content_score += 15
        reasons.append("Starker Druck zur sofortigen Zahlung bzw. riskante Zahlungsarten.")

    words = [w for w in (data.text or "").split() if len(w) > 10]
    if len(words) > 80:
        content_score += 10
        reasons.append("Viele ungewöhnlich lange Wörter – mögliche Übersetzungs-/Qualitätsprobleme.")

    if data.hasPassword:
        form_score += 10
        reasons.append("Seite fragt nach einem Passwort.")
    if data.hasCreditCard:
        form_score += 25
        reasons.append("Hinweise auf Kreditkartendaten im Inhalt gefunden.")

    base_score = url_score + content_score + form_score

    llm_result = analyze_with_llm(data.text or "")

    llm_label: Optional[str] = None
    llm_confidence: Optional[float] = None
    llm_page_type: Optional[str] = None
    frames: Optional[List[FrameInfo]] = None
    llm_extra_score = 0

    if llm_result:
        llm_label = str(llm_result.get("overall_label", "")).lower()
        llm_confidence = float(llm_result.get("overall_confidence", 0.0))
        llm_page_type = llm_result.get("page_type")

        if llm_label == "scam":
            llm_extra_score += int(40 + 40 * llm_confidence)
        elif llm_label == "suspicious":
            llm_extra_score += int(20 + 30 * llm_confidence)

        reasons.append(f"LLM-Einschätzung: {llm_label.upper()} (Confidence {llm_confidence:.2f}).")

        for rr in (llm_result.get("risk_reasons") or []):
            factor = rr.get("factor", "other")
            conf = rr.get("confidence", 0)
            expl = (rr.get("explanation", "") or "").strip()
            if expl:
                reasons.append(f"LLM-Risiko ({factor}, {conf:.2f}): {expl}")

        raw_frames = llm_result.get("frames") or []
        f_list: List[FrameInfo] = []
        for fr in raw_frames[:10]:
            trigger = (fr.get("trigger") or "").strip()
            frame_label = (fr.get("frame_label") or "").strip()
            explanation = (fr.get("explanation") or "").strip()
            if trigger and frame_label and explanation:
                f_list.append(FrameInfo(trigger=trigger, frame_label=frame_label, explanation=explanation))
        if f_list:
            frames = f_list

    total_score = base_score + llm_extra_score
    total_score = max(0, min(100, total_score))

    if total_score <= 25:
        level = "green"
    elif total_score <= 55:
        level = "yellow"
        if llm_label == "scam" and total_score < 40:
            total_score = 40
    else:
        level = "red"

    if not reasons:
        reasons.append("Keine klaren Risikosignale gemäß aktueller Heuristik erkannt (MVP).")

    return RiskResult(
        status="done",
        fromCache=False,
        url=data.url,
        score=int(total_score),
        level=level,
        reasons=reasons,
        llm_label=llm_label,
        llm_confidence=llm_confidence,
        llm_page_type=llm_page_type,
        frames=frames,
    )

# ------------------------------------------------------------------------------
# API
# ------------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "FakeSite Detector Backend läuft.", "docs": "/docs"}

@app.get("/cached", response_model=RiskResult)
def cached(url: str):
    """
    Nur bestätigter, widerspruchsfreier Cache -> sonst 404.
    """
    if not (AIRTABLE_TOKEN and AIRTABLE_BASE_ID and AIRTABLE_TABLE_NAME):
        raise HTTPException(status_code=404, detail="Caching not configured")

    url_key = url_to_key(url)
    summary = airtable_find_summary_record(url_key)

    if not summary or not should_skip_llm_using_summary(summary):
        raise HTTPException(status_code=404, detail="No confirmed cached result")

    return result_from_summary(summary, page_url=url, cached=True)

@app.post("/analyze", response_model=RiskResult)
def analyze(page: PageData):
    """
    Cache-Flow:
    - Wenn confirmed + widerspruchsfrei => cached Result (fromCache=True)
    - sonst => neue Analyse + Summary upsert
    """
    url_key = url_to_key(page.url)

    if AIRTABLE_TOKEN and AIRTABLE_BASE_ID and AIRTABLE_TABLE_NAME:
        summary = airtable_find_summary_record(url_key)
        if should_skip_llm_using_summary(summary):
            return result_from_summary(summary, page_url=page.url, cached=True)

    result = analyze_page(page)

    if AIRTABLE_TOKEN and AIRTABLE_BASE_ID and AIRTABLE_TABLE_NAME:
        airtable_upsert_summary(url_key=url_key, page_url=page.url, result=result)

    return result

@app.post("/feedback")
def receive_feedback(fb: Feedback):
    """
    1) Feedback-History-Row speichern (alte Spalten bleiben!)
    2) Summary-Row updaten (Confirmed/Counts)
    """
    if not AIRTABLE_TOKEN:
        raise RuntimeError("AIRTABLE_TOKEN ist nicht gesetzt")

    base_url = get_airtable_base_url()
    headers = airtable_headers()

    reasons_text = "\n".join(fb.backend_reasons or [])
    url_key = url_to_key(fb.url)

    payload_feedback = {
        "records": [
            {
                "fields": {
                    # ✅ alte Spalten unverändert
                    "URL": fb.url,
                    "Backend Score": fb.backend_score,
                    "Backend Level": fb.backend_level,
                    "Backend Reasons": reasons_text,
                    "User Label": fb.user_label,

                    # ✅ neue Spalten
                    "Record Type": "feedback",
                    "URL Key": url_key,
                    "Last Updated": utc_iso_z(),
                }
            }
        ]
    }

    resp = requests.post(base_url, headers=headers, json=payload_feedback, timeout=15)
    if not resp.ok:
        raise RuntimeError(f"Airtable-Fehler (Feedback row): {resp.status_code} {resp.text}")

    # Summary-Counts/Confirmed updaten (wenn Summary existiert)
    try:
        airtable_update_summary_feedback(url_key=url_key, user_label=fb.user_label)
    except Exception as e:
        print("Summary feedback update error:", repr(e))

    return {"status": "ok"}
