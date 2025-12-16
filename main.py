from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from urllib.parse import urlparse, quote

import os
import requests
import json
import re
from datetime import datetime
from typing import Optional, Literal, List, Any, Dict, Tuple

from openai import OpenAI

# ------------------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------------------

app = FastAPI(
    title="FakeSite Detector Backend",
    version="0.7",
    description="Website Security Risk + Website-Zweck/Analyse + Textanalyse (sprachliche Bilder/Frames)."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP ok
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------------------------

AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "Feedback")
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ------------------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------------------

# ✅ Website-Analyse: KEINE Frames, KEINE Evidence. Nur Zweck/Summary/Tone + Risiko-Items.
PAGE_SYSTEM_PROMPT = """
Du analysierst eine Website anhand von Kontext + Textchunks.

Ziele:
1) Beschreibe kurz, was die Seite vermutlich ist und wozu sie dient.
2) Bestimme grob page_type (shop/bank/crypto/giveaway/login/info/other) und purpose (freier Text, kurz).
3) Ton-Analyse (urgency/threat/greed_appeal) als Zahlen 0..1.
4) Sicherheits-/Scam-Risiko-Label safe/suspicious/scam anhand von Textsignalen (wenn journalistisch/informativ und unkritisch: safe/info).

Wichtig:
- Erfinde nichts. Wenn du dir unsicher bist, senke confidence.
- Gib NUR JSON aus (keinen Zusatztext, kein Markdown).

Schema:
{
  "overall_label": "safe" | "suspicious" | "scam",
  "overall_confidence": float,
  "page_type": "shop" | "bank" | "crypto" | "giveaway" | "login" | "info" | "other",
  "purpose": string,
  "summary": string,
  "tone": { "urgency": float, "threat": float, "greed_appeal": float },
  "asks_for": { "money": bool, "credentials": bool, "personal_data": bool, "crypto_transfer": bool },
  "risk_reasons": [ { "factor": string, "confidence": float, "explanation": string } ]
}

Limits:
- risk_reasons max 8
"""

# ✅ Textanalyse: Frames = sprachliche Bilder + Intent
TEXT_SYSTEM_PROMPT = """
Du analysierst einen vom Nutzer gelieferten Text (Selection oder eingefügt).
Fokus: sprachliche Bilder (Metaphern, Bildsprache, Vergleiche) und ihre Wirkung.

Ziele:
1) Kurze Zusammenfassung (1–3 Sätze): Worum geht es, und welches Framing dominiert?
2) Identifiziere sprachliche Bilder/Frames (Deutungsrahmen).
3) Für jedes Frame erkläre:
   - Image: welches Bild / welche mentale Szene erzeugt wird
   - Intent: welche Wirkung beabsichtigt ist (z.B. Angst/Empörung/Überlegenheit/Schuld/Dringlichkeit/Delegitimierung)
4) Ton (urgency/threat/greed_appeal) als Zahlen 0..1.
5) Optional: overall_label safe/suspicious/scam nur, wenn wirklich Betrug/Manipulation/Phishing/Scam-Claims im Text vorkommen.
   Reine politische/journalistische Rhetorik ist i.d.R. safe oder suspicious (bei stark manipulativer Bildsprache).

Sehr wichtig:
- Erfinde nichts.
- frames[].trigger MUSS ein exakter Substring aus dem Text sein (copy/paste), max. 8 Wörter.
- Gib NUR JSON aus.

Schema:
{
  "overall_label": "safe" | "suspicious" | "scam",
  "overall_confidence": float,
  "summary": string,
  "tone": { "urgency": float, "threat": float, "greed_appeal": float },
  "frames": [ { "trigger": string, "frame_label": string, "explanation": string } ]
}

Regeln:
- frames[].explanation MUSS enthalten: "Image: ..." und "Intent: ..."
- frames max 10
"""

# ------------------------------------------------------------------------------
# Airtable Helpers (Feedback bleibt wie gehabt)
# ------------------------------------------------------------------------------

def utc_iso_z() -> str:
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
    try:
        u = urlparse(url)
        scheme = u.scheme or "https"
        netloc = (u.netloc or "").lower()
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
        return None
    records = resp.json().get("records", [])
    return records[0] if records else None

def should_skip_llm_using_summary(summary_record: Optional[Dict[str, Any]]) -> bool:
    if not summary_record:
        return False
    f = summary_record.get("fields", {})
    confirmed = bool(f.get("Confirmed OK", False))
    fp = int(f.get("False Positive Count", 0) or 0)
    sc = int(f.get("Scam Count", 0) or 0)
    has_cached = ("Last Score" in f) and ("Last Level" in f) and ("Last Reasons" in f)
    return confirmed and fp == 0 and sc == 0 and has_cached

# ------------------------------------------------------------------------------
# Modelle
# ------------------------------------------------------------------------------

class PageData(BaseModel):
    url: str
    text: str = ""
    hasPassword: bool = False
    hasCreditCard: bool = False

class ToneInfo(BaseModel):
    urgency: float = 0.0
    threat: float = 0.0
    greed_appeal: float = 0.0

class FrameInfo(BaseModel):
    trigger: str
    frame_label: str
    explanation: str

class RiskResult(BaseModel):
    status: Literal["none", "pending", "done", "error"] = "done"
    fromCache: Optional[bool] = None
    url: Optional[str] = None

    score: int
    level: Literal["green", "yellow", "red"]
    reasons: List[str]  # ✅ hier: nur LLM-Risiken (keine Evidence)

    purpose: Optional[str] = None
    summary: Optional[str] = None
    tone: Optional[ToneInfo] = None

    llm_label: Optional[str] = None
    llm_confidence: Optional[float] = None
    llm_page_type: Optional[str] = None

class Feedback(BaseModel):
    url: str
    backend_score: int
    backend_level: str
    backend_reasons: List[str]
    user_label: Literal["ok", "scam", "false_positive"]

class TextAnalyzeRequest(BaseModel):
    url: str = ""
    text: str
    mode: Literal["selection", "custom"] = "custom"

class TextAnalyzeResult(BaseModel):
    status: Literal["none", "pending", "done", "error"] = "done"
    mode: Literal["selection", "custom"] = "custom"

    overall_label: Literal["safe", "suspicious", "scam"] = "safe"
    overall_confidence: float = 0.0
    score: int = 0

    summary: str = ""
    tone: ToneInfo = ToneInfo()
    frames: List[FrameInfo] = []

# ------------------------------------------------------------------------------
# Summary <-> RiskResult (für confirmed Airtable cache, optional)
# ------------------------------------------------------------------------------
def result_from_summary(summary_record: Dict[str, Any], page_url: str, cached: bool) -> RiskResult:
    f = summary_record.get("fields", {})
    reasons_text = f.get("Last Reasons", "") or ""
    reasons = [r for r in reasons_text.split("\n") if r.strip()]

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
        reasons=reasons or ["(Cache) Keine LLM-Risiken gespeichert."],
        llm_label=llm_label,
        llm_confidence=llm_confidence,
        llm_page_type=llm_page_type,
        purpose=(f.get("Last Purpose") or "").strip() or None,
        summary=(f.get("Last Summary") or "").strip() or None,
        tone=None,
    )

def airtable_upsert_summary(url_key: str, page_url: str, result: RiskResult) -> None:
    base_url = get_airtable_base_url()
    headers = airtable_headers()

    existing = airtable_find_summary_record(url_key)

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

        "Last LLM Label": (result.llm_label or ""),
        "Last LLM Confidence": float(result.llm_confidence or 0.0),
        "Last LLM Page Type": (result.llm_page_type or ""),

        "Last Purpose": (result.purpose or ""),
        "Last Summary": (result.summary or ""),

        "Last Updated": utc_iso_z(),
        "URL": page_url,
    }

    if existing:
        rec_id = existing["id"]
        patch_url = f"{base_url}/{rec_id}"
        requests.patch(patch_url, headers=headers, json={"fields": fields}, timeout=15)
    else:
        payload = {"records": [{"fields": fields}]}
        requests.post(base_url, headers=headers, json=payload, timeout=15)

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
    requests.patch(patch_url, headers=headers, json={"fields": patch_fields}, timeout=15)

# ------------------------------------------------------------------------------
# LLM helpers
# ------------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def split_context_and_main(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    m = re.search(r"\bMAIN TEXT:\s*", text)
    if not m:
        sep = text.find("----")
        if sep != -1:
            return text[:sep].strip(), text[sep+4:].strip()
        return "", text.strip()
    return text[:m.start()].strip(), text[m.end():].strip()

def chunk_text(text: str, chunk_size: int, overlap: int = 500, max_chunks: int = 3) -> List[str]:
    t = normalize_text(text)
    if not t:
        return []
    chunks: List[str] = []
    start = 0
    n = len(t)
    while start < n and len(chunks) < max_chunks:
        end = min(n, start + chunk_size)
        chunks.append(t[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _label_rank(label: str) -> int:
    return {"safe": 0, "suspicious": 1, "scam": 2}.get((label or "").lower(), 0)

def _dedup_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _json_from_completion(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}")
    raw_json = raw[start:end+1] if start != -1 and end != -1 else raw
    return json.loads(raw_json)

def llm_call(system_prompt: str, user_text: str) -> Optional[dict]:
    if not client or not OPENAI_API_KEY:
        return None
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        )
        raw = completion.choices[0].message.content or ""
        parsed = _json_from_completion(raw)
        # clamp conf
        parsed["overall_confidence"] = float(max(0.0, min(1.0, _safe_float(parsed.get("overall_confidence", 0.0)))))
        return parsed
    except Exception as e:
        print("LLM error:", repr(e))
        return None

def aggregate_chunk_results(items: List[dict]) -> Optional[dict]:
    if not items:
        return None

    best = None
    for r in items:
        lab = str(r.get("overall_label", "safe")).lower()
        conf = _safe_float(r.get("overall_confidence", 0.0))
        cand = (_label_rank(lab), conf, r)
        if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
            best = cand

    worst_rank, worst_conf, worst_r = best

    # page_type: prefer worst else most common
    page_type = (worst_r.get("page_type") or "").strip() or None
    if not page_type:
        counts: Dict[str, int] = {}
        for r in items:
            pt = (r.get("page_type") or "").strip()
            if pt:
                counts[pt] = counts.get(pt, 0) + 1
        if counts:
            page_type = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]

    # purpose/summary: prefer worst else longest
    purpose = (worst_r.get("purpose") or "").strip()
    summary = (worst_r.get("summary") or "").strip()
    if not purpose:
        for r in items:
            p = (r.get("purpose") or "").strip()
            if len(p) > len(purpose):
                purpose = p
    if not summary:
        for r in items:
            s = (r.get("summary") or "").strip()
            if len(s) > len(summary):
                summary = s

    # tone max
    tone = {"urgency": 0.0, "threat": 0.0, "greed_appeal": 0.0}
    asks_for = {"money": False, "credentials": False, "personal_data": False, "crypto_transfer": False}
    for r in items:
        t = r.get("tone") or {}
        tone["urgency"] = max(tone["urgency"], _safe_float(t.get("urgency", 0.0)))
        tone["threat"] = max(tone["threat"], _safe_float(t.get("threat", 0.0)))
        tone["greed_appeal"] = max(tone["greed_appeal"], _safe_float(t.get("greed_appeal", 0.0)))

        a = r.get("asks_for") or {}
        for k in asks_for.keys():
            asks_for[k] = asks_for[k] or bool(a.get(k, False))

    # risk_reasons merge + dedup
    merged_rr: List[dict] = []
    seen_rr = set()
    for r in items:
        for rr in (r.get("risk_reasons") or [])[:8]:
            if not isinstance(rr, dict):
                continue
            expl = (rr.get("explanation") or "").strip()
            factor = (rr.get("factor") or "other").strip()
            conf = _safe_float(rr.get("confidence", 0.0))
            if not expl:
                continue
            key = _dedup_key(f"{factor}::{expl}")
            if key in seen_rr:
                continue
            seen_rr.add(key)
            merged_rr.append({"factor": factor, "confidence": conf, "explanation": expl})
    merged_rr.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
    merged_rr = merged_rr[:8]

    return {
        "overall_label": "scam" if worst_rank == 2 else ("suspicious" if worst_rank == 1 else "safe"),
        "overall_confidence": float(max(0.0, min(1.0, worst_conf))),
        "page_type": page_type or "other",
        "purpose": purpose,
        "summary": summary,
        "tone": tone,
        "asks_for": asks_for,
        "risk_reasons": merged_rr
    }

def analyze_page_with_llm(page_text: str) -> Optional[dict]:
    if not client or not OPENAI_API_KEY:
        return None

    ctx_raw, main_raw = split_context_and_main(page_text)
    context_prefix = normalize_text(ctx_raw)
    main_text = main_raw or ""
    if not normalize_text(main_text):
        return None

    ctx_len = len(context_prefix)
    chunk_size = max(1600, 3600 - ctx_len)

    chunks = chunk_text(main_text, chunk_size=chunk_size, overlap=500, max_chunks=3)
    if not chunks:
        return None

    results: List[dict] = []
    for ch in chunks:
        user_text = (
            "KONTEXT:\n" + context_prefix + "\n\n"
            "CHUNK-TEXT:\n" + ch
        )
        r = llm_call(PAGE_SYSTEM_PROMPT, user_text)
        if r:
            results.append(r)

    return aggregate_chunk_results(results)

# Text frames hardening: trigger max 8 words and must appear in text
def _trigger_ok(trigger: str) -> bool:
    t = (trigger or "").strip()
    if not t:
        return False
    if len(t.split()) > 8:
        return False
    return True

def analyze_text_with_llm(text: str) -> Optional[dict]:
    t = normalize_text(text)
    if not t:
        return None
    r = llm_call(TEXT_SYSTEM_PROMPT, "TEXT:\n" + t)
    if not r:
        return None

    # harden frames
    frames = []
    for fr in (r.get("frames") or [])[:10]:
        if not isinstance(fr, dict):
            continue
        trig = (fr.get("trigger") or "").strip()
        fl = (fr.get("frame_label") or "").strip()
        ex = (fr.get("explanation") or "").strip()
        if not (trig and fl and ex):
            continue
        if not _trigger_ok(trig):
            continue
        if trig not in t:
            # try case-insensitive containment
            if trig.lower() not in t.lower():
                continue
        if "Image:" not in ex or "Intent:" not in ex:
            # enforce structure
            continue
        frames.append({"trigger": trig, "frame_label": fl, "explanation": ex})
    r["frames"] = frames
    return r

# ------------------------------------------------------------------------------
# Heuristics (für Score, aber NICHT in reasons anzeigen)
# ------------------------------------------------------------------------------

def analyze_page_heuristics(data: PageData) -> Tuple[int, List[str]]:
    url_score = 0
    content_score = 0
    form_score = 0
    internal_reasons: List[str] = []

    parsed = urlparse(data.url)
    hostname = parsed.hostname or ""

    if len(hostname) > 25 or len(hostname) < 5:
        url_score += 10
        internal_reasons.append("Ungewöhnlich lange oder sehr kurze Domain.")

    risky_tlds = [".top", ".shop", ".xyz", ".online", ".club"]
    if any(hostname.endswith(tld) for tld in risky_tlds):
        url_score += 10
        internal_reasons.append("Risky TLD.")

    if re.search(r"[0-9-]{5,}", hostname):
        url_score += 10
        internal_reasons.append("Viele Zahlen/Bindestriche in Domain.")

    if not data.url.startswith("https://"):
        url_score += 25
        internal_reasons.append("Kein HTTPS.")

    text_lower = (data.text or "").lower()

    sales_patterns = [
        "nur heute", "limited offer", "jetzt zugreifen",
        "90% rabatt", "80% rabatt", "70% rabatt",
        "deal endet in", "nur für kurze zeit",
        "jetzt bestellen", "jetzt kaufen"
    ]
    if any(p in text_lower for p in sales_patterns):
        content_score += 20
        internal_reasons.append("Aggressive Sales/Pressure.")

    phishing_patterns = [
        "ihr konto wird gesperrt",
        "bestätigen sie ihre zugangsdaten",
        "verifizieren sie ihr konto",
        "melden sie sich erneut an",
        "aus sicherheitsgründen müssen sie"
    ]
    if any(p in text_lower for p in phishing_patterns):
        content_score += 30
        internal_reasons.append("Phishing Phrasen.")

    payment_patterns = [
        "nur vorkasse", "vorkasse", "sofortüberweisung",
        "zahlen sie jetzt", "sofort bezahlen",
        "überweisung innerhalb von 24 stunden"
    ]
    if any(p in text_lower for p in payment_patterns):
        content_score += 15
        internal_reasons.append("Riskante Zahlung/Bezahldruck.")

    if data.hasPassword:
        form_score += 10
        internal_reasons.append("Passwort-Feld vorhanden.")
    if data.hasCreditCard:
        form_score += 25
        internal_reasons.append("CreditCard-Signale.")

    base_score = max(0, min(100, url_score + content_score + form_score))
    return base_score, internal_reasons

# ------------------------------------------------------------------------------
# Page Analyze
# ------------------------------------------------------------------------------

def analyze_page(data: PageData) -> RiskResult:
    base_score, _internal = analyze_page_heuristics(data)
    llm = analyze_page_with_llm(data.text or "")

    llm_label = None
    llm_conf = None
    llm_page_type = None
    purpose = None
    summary = None
    tone_obj: Optional[ToneInfo] = None

    # ✅ reasons shown in UI: ONLY LLM risks
    display_reasons: List[str] = []

    extra = 0
    if llm:
        llm_label = str(llm.get("overall_label", "")).lower()
        llm_conf = float(llm.get("overall_confidence", 0.0))
        llm_page_type = (llm.get("page_type") or "").strip() or None
        purpose = (llm.get("purpose") or "").strip() or None
        summary = (llm.get("summary") or "").strip() or None

        t = llm.get("tone") or {}
        tone_obj = ToneInfo(
            urgency=float(max(0.0, min(1.0, _safe_float(t.get("urgency", 0.0))))),
            threat=float(max(0.0, min(1.0, _safe_float(t.get("threat", 0.0))))),
            greed_appeal=float(max(0.0, min(1.0, _safe_float(t.get("greed_appeal", 0.0)))))
        )

        if llm_label == "scam":
            extra += int(40 + 40 * llm_conf)
        elif llm_label == "suspicious":
            extra += int(20 + 30 * llm_conf)

        # ✅ only risk_reasons
        for rr in (llm.get("risk_reasons") or [])[:8]:
            factor = (rr.get("factor") or "other").strip()
            conf = _safe_float(rr.get("confidence", 0.0))
            expl = (rr.get("explanation") or "").strip()
            if expl:
                display_reasons.append(f"{expl} (Factor: {factor}, {conf:.2f})")

    total = max(0, min(100, base_score + extra))

    if total <= 25:
        level = "green"
    elif total <= 55:
        level = "yellow"
        if llm_label == "scam" and total < 40:
            total = 40
    else:
        level = "red"

    if not display_reasons:
        display_reasons = ["Keine spezifischen LLM-Risiken erkannt."]

    return RiskResult(
        status="done",
        fromCache=False,
        url=data.url,
        score=int(total),
        level=level,
        reasons=display_reasons,
        purpose=purpose,
        summary=summary,
        tone=tone_obj,
        llm_label=llm_label,
        llm_confidence=llm_conf,
        llm_page_type=llm_page_type
    )

# ------------------------------------------------------------------------------
# Text Analyze (Frames/Sprachbilder)
# ------------------------------------------------------------------------------

def score_from_text_label(label: str, conf: float) -> int:
    label = (label or "safe").lower()
    conf = max(0.0, min(1.0, conf))
    if label == "scam":
        return int(65 + 35 * conf)
    if label == "suspicious":
        return int(25 + 45 * conf)
    return int(5 + 15 * (1.0 - conf))

def analyze_text(req: TextAnalyzeRequest) -> TextAnalyzeResult:
    llm = analyze_text_with_llm(req.text)
    if not llm:
        return TextAnalyzeResult(status="error", mode=req.mode, overall_label="safe", reasons=["LLM nicht verfügbar oder Text leer."])  # type: ignore

    label = str(llm.get("overall_label", "safe")).lower()
    if label not in ("safe", "suspicious", "scam"):
        label = "safe"
    conf = float(_safe_float(llm.get("overall_confidence", 0.0)))

    score = score_from_text_label(label, conf)
    summary = (llm.get("summary") or "").strip()

    t = llm.get("tone") or {}
    tone = ToneInfo(
        urgency=float(max(0.0, min(1.0, _safe_float(t.get("urgency", 0.0))))),
        threat=float(max(0.0, min(1.0, _safe_float(t.get("threat", 0.0))))),
        greed_appeal=float(max(0.0, min(1.0, _safe_float(t.get("greed_appeal", 0.0)))))
    )

    frames: List[FrameInfo] = []
    for fr in (llm.get("frames") or [])[:10]:
        trig = (fr.get("trigger") or "").strip()
        fl = (fr.get("frame_label") or "").strip()
        ex = (fr.get("explanation") or "").strip()
        if trig and fl and ex:
            frames.append(FrameInfo(trigger=trig, frame_label=fl, explanation=ex))

    return TextAnalyzeResult(
        status="done",
        mode=req.mode,
        overall_label=label,  # type: ignore
        overall_confidence=float(max(0.0, min(1.0, conf))),
        score=int(max(0, min(100, score))),
        summary=summary,
        tone=tone,
        frames=frames
    )

# ------------------------------------------------------------------------------
# API
# ------------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend läuft.", "docs": "/docs"}

@app.get("/cached", response_model=RiskResult)
def cached(url: str):
    """
    Nur confirmed Airtable Cache (optional). Lokal cached die Extension sowieso.
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
    url_key = url_to_key(page.url)

    if AIRTABLE_TOKEN and AIRTABLE_BASE_ID and AIRTABLE_TABLE_NAME:
        summary = airtable_find_summary_record(url_key)
        if should_skip_llm_using_summary(summary):
            return result_from_summary(summary, page_url=page.url, cached=True)

    result = analyze_page(page)

    if AIRTABLE_TOKEN and AIRTABLE_BASE_ID and AIRTABLE_TABLE_NAME:
        try:
            airtable_upsert_summary(url_key=url_key, page_url=page.url, result=result)
        except Exception:
            pass

    return result

@app.post("/analyze_text", response_model=TextAnalyzeResult)
def analyze_text_endpoint(req: TextAnalyzeRequest):
    if not req.text or len(req.text.strip()) < 10:
        return TextAnalyzeResult(status="error", mode=req.mode, frames=[], summary="Text ist zu kurz.")  # type: ignore
    return analyze_text(req)

@app.post("/feedback")
def receive_feedback(fb: Feedback):
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
                    "URL": fb.url,
                    "Backend Score": fb.backend_score,
                    "Backend Level": fb.backend_level,
                    "Backend Reasons": reasons_text,
                    "User Label": fb.user_label,

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

    try:
        airtable_update_summary_feedback(url_key=url_key, user_label=fb.user_label)
    except Exception:
        pass

    return {"status": "ok"}
