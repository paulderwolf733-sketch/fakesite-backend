from fastapi import FastAPI
from pydantic import BaseModel
from urllib.parse import urlparse
from typing import Literal

app = FastAPI(
    title="FakeSite Detector Backend",
    version="0.1",
    description="Einfaches Heuristik-Backend für die Risikoanalyse von Webseiten."
)

# ---- Datenmodelle ----

class PageData(BaseModel):
    url: str
    text: str = ""
    hasPassword: bool = False
    hasCreditCard: bool = False


class RiskResult(BaseModel):
    score: int
    level: str
    reasons: list[str]


class Feedback(BaseModel):
    url: str
    backend_score: int
    backend_level: str
    backend_reasons: list[str]
    user_label: Literal["ok", "scam", "false_positive"]
    

# ---- Analysefunktion ----

def analyze_page(data: PageData) -> RiskResult:
    score = 0
    reasons: list[str] = []

    # --- URL / Domain-Analyse ---
    parsed = urlparse(data.url)
    hostname = parsed.hostname or ""

    # URL-Teilscore
    url_score = 0

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
    import re
    if re.search(r"[0-9-]{5,}", hostname):
        url_score += 10
        reasons.append("Viele Zahlen oder Bindestriche in der Domain.")

    # 4) Keine HTTPS-Verbindung
    if not data.url.startswith("https://"):
        url_score += 25
        reasons.append("Seite nutzt kein HTTPS (unsichere Verbindung).")

    # --- Inhaltsanalyse (Text) ---
    content_score = 0
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
    words = [w for w in data.text.split() if len(w) > 10]
    if len(words) > 80:
        content_score += 10
        reasons.append("Viele ungewöhnlich lange Wörter – mögliche Übersetzungs-/Qualitätsprobleme.")

    # --- Formular-Flags ---
    form_score = 0
    if data.hasPassword:
        form_score += 10
        reasons.append("Seite fragt nach einem Passwort.")
    if data.hasCreditCard:
        form_score += 25
        reasons.append("Hinweise auf Kreditkartendaten im Inhalt gefunden.")

    # --- Gesamtscore zusammensetzen ---
    score = url_score + content_score + form_score

    # Obergrenze
    score = min(score, 100)

    # --- Score → Level ---
    if score <= 25:
        level = "green"
    elif score <= 55:
        level = "yellow"
    else:
        level = "red"

    if not reasons:
        reasons.append("Keine klaren Risikosignale gemäß aktueller Heuristik erkannt (MVP).")

    return RiskResult(score=score, level=level, reasons=reasons)

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
    Nimmt Feedback aus der Extension entgegen und schreibt es in eine Log-Datei.
    Später kann man damit ein ML-Modell trainieren oder Regeln verbessern.
    """
    import json
    from datetime import datetime

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "url": fb.url,
        "backend_score": fb.backend_score,
        "backend_level": fb.backend_level,
        "backend_reasons": fb.backend_reasons,
        "user_label": fb.user_label,
    }

    # Sehr simples Logging in eine JSONL-Datei
    with open("feedback_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"status": "ok"}

