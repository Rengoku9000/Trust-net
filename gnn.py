"""
gnn.py — TrustNet Phase 6: Graph-Based Network Analysis
---------------------------------------------------------
Extracts financial entities from document text, builds a directed transaction
graph with networkx, then runs structural anomaly detection:

  • Circular flows         (A→B→C→A) — strong money-laundering signal
  • Hub-and-spoke          (one entity fans out to 4+ distinct receivers)
  • Structuring / smurfing (amounts clustered just below reporting thresholds)
  • Duplicate amounts      (same amount 4+ times — template / copy-paste fraud)
  • Self-dealing           (sender == receiver in same transaction)
  • High velocity          (many transactions sharing an identical date)

Score 0-100.  Higher = more suspicious.
"""

import io
import re
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional

import networkx as nx

# ── Constants ─────────────────────────────────────────────────────────────────

# USD-equivalent thresholds where "structuring" (smurfing) is common
_THRESHOLDS      = [10_000, 5_000, 3_000]
_STRUCT_MARGIN   = 0.12      # flag amounts within 12 % below a threshold
_HUB_OUT_DEGREE  = 4         # min out-edges to call an entity a "hub"
_DUP_MIN_COUNT   = 4         # min repetitions to flag duplicate amounts
_VELOCITY_MIN    = 4         # min transactions on the same date to flag

FLAG_WEIGHTS: Dict[str, int] = {
    "circular_flow":         45,
    "structuring_detected":  35,
    "hub_and_spoke":         30,
    "self_dealing":          25,
    "duplicate_amounts":     25,
    "high_velocity":         15,
    "no_entities_found":     10,
}

# ── Text extraction (mirrors nlp.py to avoid circular imports) ────────────────

def _extract_text(content: bytes, filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "pdf":
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception:
            return ""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            decoded = content.decode(enc)
            if len(decoded) < 10:
                continue
            printable = sum(1 for c in decoded if c.isprintable() or c in "\n\r\t")
            if printable / max(len(decoded), 1) >= 0.60:
                return decoded
        except (UnicodeDecodeError, ValueError):
            continue
    return ""

# ── Entity extraction ─────────────────────────────────────────────────────────

# Matches structured account/entity codes: ACCT-1234, TXN98765, ACC12345, etc.
_RE_IDENT = re.compile(
    r"\b(?:"
    r"(?:ACCT?|A/?C|REF|TXN|TRANS|INV|CHECK|CHQ|WIRE|SRC|DST)[-#:\s]?\d{4,}"
    r"|\d{8,16}"                   # raw account numbers (8-16 digits)
    r"|[A-Z]{2,5}\d{4,}"           # coded IDs: AC1234, TXID98765
    r")\b",
    re.IGNORECASE,
)

# Dollar / currency amounts
_RE_AMOUNT = re.compile(r"[$£€]?\s*([\d]{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\b")

# Date patterns (many formats)
_RE_DATE = re.compile(
    r"\b(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}|\w+ \d{1,2},\s*\d{4})\b"
)

# Transaction pair extraction: "from X to Y $Z" and "X → Y $Z"
_RE_FROM_TO = re.compile(
    r"(?:from|sender|payer|source)[:\s]+([A-Z0-9\-#]{3,})"
    r".*?(?:to|receiver|recipient|beneficiary|payee)[:\s]+([A-Z0-9\-#]{3,})"
    r".*?[$£€]?\s*([\d,]+(?:\.\d{2})?)",
    re.IGNORECASE | re.DOTALL,
)

_RE_INLINE = re.compile(
    r"([A-Z0-9\-]{3,})\s*(?:→|->|paid|transferred to|sent to|wire to)\s*([A-Z0-9\-]{3,})"
    r".*?[$£€]?\s*([\d,]+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)


def _parse_amounts(text: str) -> List[float]:
    results = []
    for m in _RE_AMOUNT.finditer(text):
        try:
            val = float(m.group(1).replace(",", ""))
            if val >= 1:                    # ignore trivial amounts < $1
                results.append(val)
        except ValueError:
            pass
    return results


def _parse_identifiers(text: str) -> List[str]:
    return [
        m.group(0).upper().replace(" ", "").replace("#", "").replace(":", "")
        for m in _RE_IDENT.finditer(text)
    ]


def _parse_dates(text: str) -> List[str]:
    return _RE_DATE.findall(text)


def _parse_triples(text: str) -> List[Tuple[str, str, float]]:
    """Return (from_entity, to_entity, amount) triples."""
    triples: List[Tuple[str, str, float]] = []

    for m in _RE_FROM_TO.finditer(text):
        try:
            amt = float(m.group(3).replace(",", ""))
            if amt > 0:
                triples.append((m.group(1).upper(), m.group(2).upper(), amt))
        except ValueError:
            pass

    for m in _RE_INLINE.finditer(text):
        src, dst = m.group(1).upper(), m.group(2).upper()
        try:
            amt = float(m.group(3).replace(",", ""))
            if amt > 0 and src != dst:
                triples.append((src, dst, amt))
        except ValueError:
            pass

    # De-duplicate
    return list(dict.fromkeys(triples))

# ── Structural checks ─────────────────────────────────────────────────────────

def _check_structuring(amounts: List[float]) -> Tuple[bool, dict]:
    """Flag amounts that cluster just below common reporting thresholds."""
    if not amounts:
        return False, {}
    clusters = []
    for thresh in _THRESHOLDS:
        lo = thresh * (1 - _STRUCT_MARGIN)
        hits = sorted(a for a in amounts if lo <= a < thresh)
        if len(hits) >= 2:
            clusters.append({"threshold": thresh, "hits": len(hits), "amounts": hits[:5]})
    return bool(clusters), {"structuring_clusters": clusters} if clusters else {}


def _check_duplicate_amounts(amounts: List[float]) -> Tuple[bool, dict]:
    """Flag any amount appearing 4+ times (template / copy-paste fraud)."""
    if len(amounts) < _DUP_MIN_COUNT:
        return False, {}
    counts = Counter(round(a, 2) for a in amounts)
    dups = {str(k): v for k, v in counts.items() if v >= _DUP_MIN_COUNT}
    return bool(dups), {"duplicate_amounts": dups} if dups else {}


def _check_high_velocity(dates: List[str]) -> Tuple[bool, dict]:
    """Flag if many transactions share the exact same date string."""
    if not dates:
        return False, {}
    counts = Counter(dates)
    hot = {d: c for d, c in counts.items() if c >= _VELOCITY_MIN}
    return bool(hot), {"high_velocity_dates": hot} if hot else {}


def _check_hub_spoke(G: nx.DiGraph) -> Tuple[bool, dict]:
    """Flag nodes with out-degree >= threshold (fan-out hub)."""
    hubs = {n: G.out_degree(n) for n in G.nodes if G.out_degree(n) >= _HUB_OUT_DEGREE}
    return bool(hubs), {"hub_entities": hubs} if hubs else {}


def _check_cycles(G: nx.DiGraph) -> Tuple[bool, list]:
    """Detect circular flows using networkx simple_cycles."""
    try:
        cycles = list(nx.simple_cycles(G))
        return bool(cycles), [c for c in cycles[:3]]
    except Exception:
        return False, []


def _check_self_dealing(triples: List[Tuple[str, str, float]]) -> Tuple[bool, list]:
    """Flag transactions where sender == receiver."""
    self_deals = [(s, d, a) for s, d, a in triples if s == d]
    return bool(self_deals), [f"{s} -> {d}  ${a:,.2f}" for s, d, a in self_deals[:3]]

# ── Public API ────────────────────────────────────────────────────────────────

def perform_gnn(content: bytes, filename: str = "") -> Dict[str, Any]:
    """
    Main entry point.  Extract text → build entity graph → run anomaly checks.
    Returns score (0–100), flags, message, and analysis details.
    """
    text = _extract_text(content, filename)

    flags:   List[str]      = []
    details: Dict[str, Any] = {}

    if not text or not text.strip():
        flags.append("no_entities_found")
        return _build_result(flags, details)

    amounts     = _parse_amounts(text)
    identifiers = _parse_identifiers(text)
    dates       = _parse_dates(text)
    triples     = _parse_triples(text)

    details["amounts_found"]            = len(amounts)
    details["unique_identifiers_found"] = len(set(identifiers))
    details["dates_found"]              = len(dates)
    details["transaction_pairs_found"]  = len(triples)

    if not amounts and not identifiers:
        flags.append("no_entities_found")
        return _build_result(flags, details)

    # ── Amount-level checks (no graph needed) ─────────────────────────────────

    struct_flag, struct_detail = _check_structuring(amounts)
    details.update(struct_detail)
    if struct_flag:
        flags.append("structuring_detected")

    dup_flag, dup_detail = _check_duplicate_amounts(amounts)
    details.update(dup_detail)
    if dup_flag:
        flags.append("duplicate_amounts")

    vel_flag, vel_detail = _check_high_velocity(dates)
    details.update(vel_detail)
    if vel_flag:
        flags.append("high_velocity")

    # ── Graph-level checks (requires extracted transaction pairs) ─────────────

    if triples:
        G = nx.DiGraph()
        for src, dst, amt in triples:
            # Accumulate weight if edge already exists
            if G.has_edge(src, dst):
                G[src][dst]["weight"] += amt
                G[src][dst]["count"]  += 1
            else:
                G.add_edge(src, dst, weight=amt, count=1)

        details["graph_nodes"] = G.number_of_nodes()
        details["graph_edges"] = G.number_of_edges()

        hub_flag, hub_detail = _check_hub_spoke(G)
        details.update(hub_detail)
        if hub_flag:
            flags.append("hub_and_spoke")

        cycle_flag, cycles = _check_cycles(G)
        if cycle_flag:
            flags.append("circular_flow")
            details["circular_flows"] = cycles

        self_flag, self_examples = _check_self_dealing(triples)
        if self_flag:
            flags.append("self_dealing")
            details["self_dealing_examples"] = self_examples

    return _build_result(flags, details)


def _build_result(flags: List[str], details: Dict[str, Any]) -> Dict[str, Any]:
    score = min(100, sum(FLAG_WEIGHTS.get(f, 0) for f in flags))

    if score >= 70:
        message = "HIGH: Critical graph anomalies — strong indicators of structured financial fraud."
    elif score >= 30:
        message = "MEDIUM: Suspicious transaction patterns — further investigation warranted."
    else:
        message = "LOW: Transaction graph appears structurally normal."

    return {"score": score, "flags": flags, "message": message, "details": details}
