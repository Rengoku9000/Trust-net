"""
nlp.py — TrustNet Phase 5: NLP Text Anomaly Scoring
------------------------------------------------------
Extracts readable text from PDFs (via pdfplumber) or plain text, then runs
a battery of linguistic / statistical checks to surface fraud signals:

  • Urgency / pressure language       ("final notice", "within 24 hours")
  • Threat language                   ("legal action", "arrest", "warrant")
  • Financial fraud patterns          ("wire transfer", "gift card", "you have won")
  • Excessive capitalisation          (shouting / emphasis abuse)
  • Copy-paste repetition             (duplicated sentences)
  • Suspicious number formatting      (all-round amounts, mixed currency styles)
  • Encoding anomalies                (replacement chars, non-printable bytes)

Score 0-100.  Higher = more suspicious.
"""

import io
import re
from collections import Counter
from typing import Dict, Any, List, Tuple

# ── Pattern libraries ─────────────────────────────────────────────────────────

_URGENCY = [
    r"\burgent\b",
    r"\bimmediately\b",
    r"\bfinal\s+notice\b",
    r"\blast\s+warning\b",
    r"\bact\s+now\b",
    r"\bdo\s+not\s+ignore\b",
    r"\bimmediate\s+action\b",
    r"\btime[\-\s]sensitive\b",
    r"\bwithin\s+\d+\s+hours?\b",
    r"\bwithin\s+\d+\s+days?\b",
    r"\boverdue\b",
    r"\bpast\s+due\b",
    r"\bdeadline\b",
    r"\bexpires?\s+(today|tomorrow|soon)\b",
]

_THREAT = [
    r"\blegal\s+action\b",
    r"\blawsuit\b",
    r"\barrest\b",
    r"\bprosecute[sd]?\b",
    r"\bcriminal\s+charges?\b",
    r"\blaw\s+enforcement\b",
    r"\bwarrant\b",
    r"\bjail\b",
    r"\bprison\b",
    r"\bpenalt(y|ies)\b",
    r"\bseize[sd]?\b",
    r"\brepossess(ed|ion)?\b",
]

_FRAUD = [
    r"\bwire\s+transfer\b",
    r"\bwestern\s+union\b",
    r"\bmoney\s+order\b",
    r"\bgift\s+cards?\b",
    r"\bitunes\s+card\b",
    r"\bgoogle\s+play\s+card\b",
    r"\byou\s+have\s+won\b",
    r"\bunclaimed\s+funds?\b",
    r"\binheritance\s+(of|worth|valued)\b",
    r"\bestate\s+of\s+the\s+late\b",
    r"\badvance\s+fee\b",
    r"\bprocessing\s+fee\s+required\b",
    r"\bconfidential\s+(fund|transfer|transaction)\b",
    r"\bverify\s+your\s+(account|information|details|identity)\b",
    r"\bupdate\s+your\s+(account|password|billing|payment)\b",
    r"\bclick\s+here\s+to\b",
    r"\bsend\s+(me|us)\s+(your\s+)?(bank|account|card|cvv)\b",
    r"\bprovide\s+(your\s+)?(ssn|social\s+security|bank\s+account)\b",
]

# Compile all patterns once at import time
def _compile(patterns: List[str]):
    return [re.compile(p, re.IGNORECASE) for p in patterns]

_RE_URGENCY = _compile(_URGENCY)
_RE_THREAT  = _compile(_THREAT)
_RE_FRAUD   = _compile(_FRAUD)

# Currency / number patterns
_RE_CURRENCY = re.compile(r"[\$£€¥]\s*[\d,]+(?:\.\d{1,2})?|\b\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?\b")
_RE_NUMBER   = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_RE_ENCODING = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# ── Flag weights ──────────────────────────────────────────────────────────────

FLAG_WEIGHTS: Dict[str, int] = {
    "financial_fraud_patterns":      40,
    "threat_language":               35,
    "high_urgency_language":         25,
    "excessive_capitalisation":      20,
    "copy_paste_repetition":         20,
    "suspicious_number_formatting":  15,
    "encoding_anomalies":            15,
    "very_short_text":               10,
    "no_text_extractable":           10,
}

# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_text(content: bytes, filename: str) -> str:
    """Return plain text from a PDF or text file; empty string on failure."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "\n".join(
                    (page.extract_text() or "") for page in pdf.pages
                )
        except Exception:
            return ""

    # Try decoding as plain text (handles .txt, .csv, unknown)
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            decoded = content.decode(enc)
            # Reject blobs that are too short or mostly non-printable (binary/image data)
            if len(decoded) < 10:
                continue
            printable = sum(1 for c in decoded if c.isprintable() or c in "\n\r\t")
            if printable / max(len(decoded), 1) >= 0.60:
                return decoded
        except (UnicodeDecodeError, ValueError):
            continue

    return ""

# ── Individual checks ─────────────────────────────────────────────────────────

def _count_pattern_hits(text: str, patterns) -> Tuple[int, List[str]]:
    """Return (total hit count, list of unique matched strings)."""
    hits, matched = 0, []
    for pat in patterns:
        found = pat.findall(text)
        if found:
            hits += len(found)
            matched.append(found[0] if isinstance(found[0], str) else found[0])
    return hits, matched


def _check_capitalisation(text: str) -> Tuple[bool, float]:
    """True if >12 % of 3+ character words are ALL-CAPS."""
    words = re.findall(r"\b[A-Za-z]{3,}\b", text)
    if not words:
        return False, 0.0
    caps_ratio = sum(1 for w in words if w.isupper()) / len(words)
    return caps_ratio > 0.12, round(caps_ratio, 3)


def _check_repetition(text: str) -> Tuple[bool, List[str]]:
    """True if any sentence appears 3+ times (copy-paste artifact)."""
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", text) if len(s.strip()) > 20]
    counts = Counter(sentences)
    repeated = [s for s, c in counts.items() if c >= 3]
    return bool(repeated), repeated[:3]  # cap at 3 examples


def _check_number_formatting(text: str) -> Tuple[bool, dict]:
    """
    Flags if:
    - All numeric values are suspiciously round (no cents in a financial doc)
    - Multiple conflicting currency styles appear
    """
    currency_amounts = _RE_CURRENCY.findall(text)
    all_numbers      = _RE_NUMBER.findall(text)

    details: Dict[str, Any] = {
        "currency_amounts_found": len(currency_amounts),
        "total_numbers_found":    len(all_numbers),
    }

    if len(currency_amounts) < 2:
        return False, details  # Not enough data

    # Check if every amount is a round number (no decimal / ends in .00)
    round_count = sum(
        1 for a in currency_amounts
        if not re.search(r"\.\d[1-9]", a)   # no non-zero cents
    )
    details["round_number_ratio"] = round(round_count / len(currency_amounts), 2)
    suspicious = round_count == len(currency_amounts) and len(currency_amounts) >= 3

    # Check for conflicting currency symbols
    symbols = re.findall(r"[\$£€¥]", text)
    unique_symbols = set(symbols)
    details["currency_symbols"] = list(unique_symbols)
    if len(unique_symbols) > 1:
        suspicious = True
        details["mixed_currencies"] = True

    return suspicious, details


def _check_encoding(text: str) -> Tuple[bool, int]:
    """True if replacement/control characters exceed 0.5 % of text."""
    if not text:
        return False, 0
    anomalies = len(_RE_ENCODING.findall(text))
    ratio = anomalies / len(text)
    return ratio > 0.005, anomalies

# ── Public API ────────────────────────────────────────────────────────────────

def perform_nlp(content: bytes, filename: str = "") -> Dict[str, Any]:
    """
    Main entry point.  Extract text, run all checks, return score + details.
    """
    text = _extract_text(content, filename)

    flags:   List[str]       = []
    details: Dict[str, Any]  = {}

    # ── No text ───────────────────────────────────────────────────────────────
    if not text or not text.strip():
        flags.append("no_text_extractable")
        return _build_result(flags, details)

    word_count = len(text.split())
    details["word_count"] = word_count

    # ── Very short text ───────────────────────────────────────────────────────
    if word_count < 30:
        flags.append("very_short_text")

    # ── Urgency language ──────────────────────────────────────────────────────
    urgency_hits, urgency_matches = _count_pattern_hits(text, _RE_URGENCY)
    details["urgency_hits"] = urgency_hits
    if urgency_hits >= 2:
        flags.append("high_urgency_language")
        details["urgency_examples"] = urgency_matches[:5]

    # ── Threat language ───────────────────────────────────────────────────────
    threat_hits, threat_matches = _count_pattern_hits(text, _RE_THREAT)
    details["threat_hits"] = threat_hits
    if threat_hits >= 1:
        flags.append("threat_language")
        details["threat_examples"] = threat_matches[:5]

    # ── Financial fraud patterns ──────────────────────────────────────────────
    fraud_hits, fraud_matches = _count_pattern_hits(text, _RE_FRAUD)
    details["fraud_hits"] = fraud_hits
    if fraud_hits >= 1:
        flags.append("financial_fraud_patterns")
        details["fraud_examples"] = fraud_matches[:5]

    # ── Excessive capitalisation ──────────────────────────────────────────────
    caps_flag, caps_ratio = _check_capitalisation(text)
    details["caps_ratio"] = caps_ratio
    if caps_flag:
        flags.append("excessive_capitalisation")

    # ── Copy-paste repetition ─────────────────────────────────────────────────
    rep_flag, rep_examples = _check_repetition(text)
    if rep_flag:
        flags.append("copy_paste_repetition")
        details["repeated_sentences"] = rep_examples

    # ── Number formatting ─────────────────────────────────────────────────────
    num_flag, num_details = _check_number_formatting(text)
    details["number_analysis"] = num_details
    if num_flag:
        flags.append("suspicious_number_formatting")

    # ── Encoding anomalies ────────────────────────────────────────────────────
    enc_flag, enc_count = _check_encoding(text)
    details["encoding_anomaly_chars"] = enc_count
    if enc_flag:
        flags.append("encoding_anomalies")

    return _build_result(flags, details)


def _build_result(flags: List[str], details: Dict[str, Any]) -> Dict[str, Any]:
    score = min(100, sum(FLAG_WEIGHTS.get(f, 0) for f in flags))

    if score >= 70:
        message = "HIGH: Strong textual fraud indicators — immediate review required."
    elif score >= 30:
        message = "MEDIUM: Suspicious language patterns detected — warrants investigation."
    else:
        message = "LOW: Text appears consistent with a legitimate financial document."

    return {"score": score, "flags": flags, "message": message, "details": details}
