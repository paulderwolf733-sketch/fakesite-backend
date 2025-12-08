from fastapi import FastAPI
from pydantic import BaseModel
from urllib.parse import urlparse
import os
import requests
import json
import re
from typing import Literal
from datetime import datetime
from openai import OpenAI  # neues OpenAI SDK

app = FastAPI(
    title="FakeSite Detector Backend",
    version="0.1",
    description="Einfaches Heuristik-Backend für die Risikoanalyse von Webseiten."
)

AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "Feedback")
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# OpenAI-Client nur initialisieren, wenn ein API-Key gesetzt ist
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

LLM_SYSTEM_PROMPT = """
Du bist ein automatischer Klassifikator für potenziell betrügerische Webseiten.

Du bekommst einen AUSZUG aus dem Text einer Website (max. einige tausend Zeichen).
Deine Aufgabe ist, NUR anhand dieses Textes eine Risikoeinschätzung vorzunehmen.

Ziele:
- Erkenne, ob der Text harmlos, verdächtig oder sehr wahrscheinlich Betrug ist.
- Erkenne den Zweck der Seite (Shop, Bank/Login, Gewinnspiel, Crypto/Investment, Info, Sonstiges).
- Erkenne Ton und Muster:
  - Dringlichkeit / künstlicher Zeitdruck
  - Drohungen (z.B. Konto wird gesperrt)
  - Appell an Gier („schnell reich werden“, unrealistische Gewinne)
  - Forderung nach Geld, Zugangsdaten, persönlichen Daten oder Krypto-Transaktionen.

Deine Ausgabe MUSS ein gültiges JSON-Objekt sein, OHNE zusätzlichen Text, OHNE Erklärungen, OHNE Markdown.

Das JSON MUSS GENAU dieses Schema haben:

{
  "overall_label": "safe" | "suspicious" | "scam",
  "overall_confidence": float (0.0 bis 1.0),
  "page_type": "shop" | "bank" | "crypto" | "giveaway" | "login" | "info" | "other",
  "tone": {
    "urgency": float (0.0 bis 1.0),
    "threat": float (0.0 bis 1.0),
    "greed_appeal": float (0.0 bis 1.0)
  },
  "asks_for": {
    "money": bool,
    "credentials": bool,
    "personal_data": bool,
    "crypto_transfer": bool
  },
  "risk_reasons": [
    {
      "factor": "urgency" | "threat" | "unrealistic_offer" | "payment_method" | "identity_theft" | "other",
      "confidence": float (0.0 bis 1.0),
      "explanation": string (kurze Begründung auf Deutsch)
    }
  ]
}

Hinweise:
- Wenn du KEIN klares Risiko-Ergebnis hast, verwende "safe" oder "suspicious" mit passender Confidence.
- Wenn der Text gemischt ist, bewerte die Dominanz: Wenn Scam-Elemente überwiegen, nutze "scam".
- overall_confidence beschreibt deine Sicherheit in der GESAMT-Klassifikation.
"""

def truncate_text(text: str, max_chars: int = 4000) -> str:
    """Whitespace normalisieren und Text auf max_chars kürzen."""
    if not text:
        return ""
    # Mehrfache Leerzeichen/Zeilenumbrüche zusammenschrumpfen
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars]

def analyze_with_llm(page_text: str) -> dict | None:
    """
    Ruft gpt-4.1-mini auf, um den Text zu klassifizieren.
    Gibt ein Dict mit den Feldern aus dem Schema zurück oder None bei Fehler.
    """
    if not client or not OPENAI_API_KEY:
        # Kein API-Key -> LLM wird einfach übersprungen
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
                {
                    "role": "user",
                    "content": (
                        "Hier ist der Textauszug der Website, den du analysieren sollst:\n\n"
                        f"{snippet}"
                    ),
                },
            ],
        )

        raw_content = completion.choices[0].message.content or ""

        # Versuche JSON sicher zu extrahieren (falls das Modell doch mal irgendwas drumherum schreibt)
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start != -1 and end != -1:
            raw_json = raw_content[start : end + 1]
        else:
            raw_json = raw_content

        data = json.loads(raw_json)
        return data
    except Exception as e:
        # Zum Debuggen in den Logs ansehen, aber die Analyse nicht komplett abbrechen
        print("LLM-Analysefehler:", repr(e))
        return None

def get_airtable_url() -> str:
    if not AIRTABLE_BASE_ID:
        raise RuntimeError("AIRTABLE_BASE_ID ist nicht gesetzt")
    if not AIRTABLE_TABLE_NAME:
        raise RuntimeError("AIRTABLE_TABLE_NAME ist nicht gesetzt")
    return f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"

# ---- Datenmodelle ----

class PageData(BaseModel):
    url: str
    text: str = ""
    hasPassword: bool = False
    hasCreditCard: bool = False


from typing import Optional, Literal

class RiskResult(BaseModel):
    score: int
    level: Literal["green", "yellow", "red"]
    reasons: list[str]

    # optionale Zusatzinfos aus der LLM-Analyse
    llm_label: Optional[str] = None
    llm_confidence: Optional[float] = None
    llm_page_type: Optional[str] = None


class Feedback(BaseModel):
    url: str
    backend_score: int
    backend_level: str
    backend_reasons: list[str]
    user_label: Literal["ok", "scam", "false_positive"]
    

# ---- Analysefunktion ----

def analyze_page(data: PageData) -> RiskResult:
    # --------- 1) Heuristik wie bisher ---------
    url_score = 0
    content_score = 0
    form_score = 0
    reasons: list[str] = []

    # --- URL / Domain-Analyse ---
    parsed = urlparse(data.url)
    hostname = parsed.hostname or ""

    # 1) Ungewöhnliche Länge
    if len(hostname) > 25 or len(hostname) < 5:
        url_score += 10
        reasons.append("Ungewöhnlich lange oder sehr kurze Domain.")

    # 2) Auffällige TLDs
    risky_tlds = [".top", ".shop", ".xyz", ".online", ".club"]
    if any(hostname.endswith(tld) for tld in risky_tlds):
        url_score += 10
        reasons.append("Top-Level-Domain ist häufiger bei Fake-Shops anzutreffen.")

    # 3) Viele Zahlen / Bindestriche in der Domain
    if re.search(r"[0-9-]{5,}", hostname):
        url_score += 10
        reasons.append("Viele Zahlen oder Bindestriche in der Domain.")

    # 4) Keine HTTPS-Verbindung
    if not data.url.startswith("https://"):
        url_score += 25
        reasons.append("Seite nutzt kein HTTPS (unsichere Verbindung).")

    # --- Inhaltsanalyse (Text) ---
    text_lower = (data.text or "").lower()

    # Aggressive Sales-Phrasen
    sales_patterns = [
        "nur heute", "limited offer", "jetzt zugreifen",
        "90% rabatt", "80% rabatt", "70% rabatt",
        "deal endet in", "nur für kurze zeit",
        "jetzt bestellen", "jetzt kaufen"
    ]
    if any(p in text_lower for p in sales_patterns):
        content_score += 20
        reasons.append("Aggressive Verkaufs- oder Druckformulierungen gefunden.")

    # Phishing-Phrasen
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

    # Zahlungsdruck / riskante Zahlungsarten
    payment_patterns = [
        "nur vorkasse", "vorkasse", "sofortüberweisung",
        "zahlen sie jetzt", "sofort bezahlen",
        "überweisung innerhalb von 24 stunden"
    ]
    if any(p in text_lower for p in payment_patterns):
        content_score += 15
        reasons.append("Starker Druck zur sofortigen Zahlung bzw. riskante Zahlungsarten.")

    # Grobe „Qualitäts“-Heuristik
    words = [w for w in (data.text or "").split() if len(w) > 10]
    if len(words) > 80:
        content_score += 10
        reasons.append("Viele ungewöhnlich lange Wörter – mögliche Übersetzungs-/Qualitätsprobleme.")

    # --- Formular-Flags ---
    if data.hasPassword:
        form_score += 10
        reasons.append("Seite fragt nach einem Passwort.")
    if data.hasCreditCard:
        form_score += 25
        reasons.append("Hinweise auf Kreditkartendaten im Inhalt gefunden.")

    base_score = url_score + content_score + form_score

    # --------- 2) LLM-Analyse des Textes ---------
    llm_result = analyze_with_llm(data.text or "")
    llm_label = None
    llm_confidence = None
    llm_page_type = None

    llm_extra_score = 0

    if llm_result:
        llm_label = str(llm_result.get("overall_label", "")).lower()
        llm_confidence = float(llm_result.get("overall_confidence", 0.0))
        llm_page_type = llm_result.get("page_type")

        # Score-Beitrag des LLM
        # Du kannst das Gewicht später feinjustieren.
        if llm_label == "scam":
            # starke Warnung
            llm_extra_score += int(40 + 40 * llm_confidence)  # 40–80 Punkte
        elif llm_label == "suspicious":
            llm_extra_score += int(20 + 30 * llm_confidence)  # 20–50 Punkte
        elif llm_label == "safe":
            # wir machen es nicht "negativ", sondern lassen base_score bestehen
            llm_extra_score += 0

        # LLM-Einschätzung als Begründung hinzufügen
        reasons.append(
            f"LLM-Einschätzung: {llm_label.upper()} (Confidence {llm_confidence:.2f})."
        )

        # Einzelne Risk-Faktoren ausgeben
        risk_reasons = llm_result.get("risk_reasons") or []
        for rr in risk_reasons:
            factor = rr.get("factor", "other")
            conf = rr.get("confidence", 0)
            expl = rr.get("explanation", "").strip()
            if expl:
                reasons.append(f"LLM-Risiko ({factor}, {conf:.2f}): {expl}")

    # --------- 3) Gesamtscore & Level ---------
    total_score = base_score + llm_extra_score
    if total_score < 0:
        total_score = 0
    if total_score > 100:
        total_score = 100

    if total_score <= 25:
        level = "green"
    elif total_score <= 55:
        level = "yellow"
        # optional: minimaler Score, wenn LLM "scam" sagt, aber Basiswert niedrig ist
        if llm_label == "scam" and total_score < 40:
            total_score = 40
    else:
        level = "red"

    if not reasons:
        reasons.append("Keine klaren Risikosignale gemäß aktueller Heuristik erkannt (MVP).")

    return RiskResult(
        score=total_score,
        level=level,
        reasons=reasons,
        # Die folgenden Felder kannst du in Airtable oder späterem ML nutzen
        llm_label=llm_label,
        llm_confidence=llm_confidence,
        llm_page_type=llm_page_type,
    )


# ---- API-Endpunkte ----

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "FakeSite Detector Backend läuft.",
        "docs": "/docs"
    }


@app.post("/analyze", response_model=RiskResult)
def analyze(page: PageData):
    """
    Nimmt URL, Textauszug und einfache Flags entgegen und liefert
    einen Risiko-Score (0–100), Level (green/yellow/red) und Begründungen zurück.
    """
    result = analyze_page(page)
    return result


@app.post("/feedback")
def receive_feedback(fb: Feedback):
    """
    Nimmt Feedback aus der Extension entgegen und speichert es in Airtable.
    """
    if not AIRTABLE_TOKEN:
        raise RuntimeError("AIRTABLE_TOKEN ist nicht gesetzt")

    airtable_url = get_airtable_url()

    headers = {
        "Authorization": f"Bearer {AIRTABLE_TOKEN}",
        "Content-Type": "application/json",
    }

    # Backend-Reasons in einen String packen
    reasons_text = "\n".join(fb.backend_reasons or [])

    payload = {
        "records": [
            {
                "fields": {
                    "URL": fb.url,
                    "Backend Score": fb.backend_score,
                    "Backend Level": fb.backend_level,
                    "Backend Reasons": reasons_text,
                    "User Label": fb.user_label,
                }
            }
        ]
    }

    resp = requests.post(airtable_url, headers=headers, json=payload, timeout=10)
    if not resp.ok:
        # Zum Debuggen kannst du dir das Response-JSON auch in den Logs anschauen
        raise RuntimeError(f"Airtable-Fehler: {resp.status_code} {resp.text}")

    return {"status": "ok"}


