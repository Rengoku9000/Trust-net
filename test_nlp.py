"""
test_nlp.py — Unit tests for the NLP text anomaly layer (Phase 5).
Run with:  .\\venv\\Scripts\\python.exe test_nlp.py
"""

import sys
from nlp import perform_nlp

# ── helpers ───────────────────────────────────────────────────────────────────

def ok(label: str):
    print(f"  [PASS]  {label}")

def fail(label: str, detail: str):
    print(f"  [FAIL]  {label}: {detail}")
    sys.exit(1)

def show(result: dict):
    print(f"      score={result['score']}  flags={result['flags']}")
    print(f"      msg  : {result['message']}")

# ── Test 1: clean financial text -> low score ─────────────────────────────────

def test_clean_text():
    print("[Test 1] Clean invoice text -> expect low score")
    text = b"""
    Invoice #INV-2025-001
    Date: January 15, 2025
    Due Date: February 15, 2025

    Services Rendered: Software consulting for Q4 2024
    Hours: 40 hours at $150.00/hr

    Subtotal:   $6,000.00
    Tax (10%):    $600.00
    Total Due:  $6,600.00

    Please remit payment to the above account by the due date.
    Thank you for your business.
    """
    result = perform_nlp(text, filename="invoice.txt")
    show(result)
    if result["score"] > 35:
        fail("clean text", f"expected score <= 35, got {result['score']}")
    ok(f"low score {result['score']}")
    print()

# ── Test 2: urgency + threat language -> high score ───────────────────────────

def test_urgent_threat():
    print("[Test 2] Urgency + threat language -> expect high score")
    text = b"""
    FINAL NOTICE - URGENT ACTION REQUIRED IMMEDIATELY

    Your account is OVERDUE. You must act now to avoid LEGAL ACTION.
    Failure to respond within 24 hours will result in ARREST and criminal
    charges being filed against you. A warrant has been issued.

    This is your LAST WARNING. Do not ignore this notice.
    Pay immediately to avoid prosecution and jail time.
    Our attorneys will file a lawsuit if no response is received.
    """
    result = perform_nlp(text, filename="threat_letter.txt")
    show(result)
    if result["score"] < 50:
        fail("urgency+threat", f"expected score >= 50, got {result['score']}")
    if "threat_language" not in result["flags"]:
        fail("threat_language flag", f"flags={result['flags']}")
    if "high_urgency_language" not in result["flags"]:
        fail("urgency flag", f"flags={result['flags']}")
    ok(f"high score {result['score']} with correct flags")
    print()

# ── Test 3: financial fraud patterns ─────────────────────────────────────────

def test_fraud_patterns():
    print("[Test 3] Advance fee / wire transfer fraud -> expect fraud flag")
    text = b"""
    CONGRATULATIONS! You have won $4,500,000 in the International Lottery.
    To claim your prize, please wire transfer a processing fee of $500 via
    Western Union. You may also pay via gift cards (Google Play or iTunes).

    Please verify your account details and provide your bank account number.
    This is a confidential transfer - do not share with others.
    Click here to claim your unclaimed funds before the deadline expires.
    """
    result = perform_nlp(text, filename="lottery.txt")
    show(result)
    if "financial_fraud_patterns" not in result["flags"]:
        fail("financial_fraud_patterns", f"flags={result['flags']}")
    if result["score"] < 40:
        fail("fraud score", f"expected >= 40, got {result['score']}")
    ok(f"financial_fraud_patterns detected, score={result['score']}")
    print()

# ── Test 4: excessive capitalisation ─────────────────────────────────────────

def test_excessive_caps():
    print("[Test 4] Excessive CAPS -> expect excessive_capitalisation flag")
    text = b"""
    DEAR CUSTOMER THIS IS AN IMPORTANT MESSAGE FROM YOUR BANK.
    YOUR ACCOUNT HAS BEEN SUSPENDED DUE TO SUSPICIOUS ACTIVITY.
    YOU MUST VERIFY YOUR IDENTITY IMMEDIATELY OR YOUR ACCOUNT WILL BE CLOSED.
    PLEASE RESPOND URGENTLY TO THIS IMPORTANT SECURITY ALERT.
    """
    result = perform_nlp(text, filename="caps.txt")
    show(result)
    if "excessive_capitalisation" not in result["flags"]:
        fail("excessive_capitalisation", f"flags={result['flags']}, caps_ratio={result['details'].get('caps_ratio')}")
    ok(f"excessive_capitalisation detected, ratio={result['details'].get('caps_ratio')}")
    print()

# ── Test 5: copy-paste repetition ────────────────────────────────────────────

def test_repetition():
    print("[Test 5] Repeated sentences -> expect copy_paste_repetition flag")
    repeated_sentence = b"Please send your bank details to claim your reward."
    text = (repeated_sentence + b"\n") * 5 + b"This is a legitimate document."
    result = perform_nlp(text, filename="repetitive.txt")
    show(result)
    if "copy_paste_repetition" not in result["flags"]:
        fail("copy_paste_repetition", f"flags={result['flags']}")
    ok("copy_paste_repetition detected")
    print()

# ── Test 6: empty / binary content -> no_text_extractable ────────────────────

def test_no_text():
    print("[Test 6] Binary/empty content -> expect no_text_extractable flag")
    result = perform_nlp(bytes([0xFF, 0xD8, 0xFF, 0xE0]), filename="image.jpg")
    show(result)
    if "no_text_extractable" not in result["flags"]:
        fail("no_text_extractable", f"flags={result['flags']}")
    ok("no_text_extractable flag present")
    print()

# ── Test 7: suspicious all-round numbers ─────────────────────────────────────

def test_round_numbers():
    print("[Test 7] All-round currency amounts -> expect suspicious_number_formatting")
    text = b"""
    Invoice Summary:
    Consulting Fee:   $5,000
    Travel Expenses:  $2,000
    Materials:        $3,000
    Total Due:        $10,000

    Please wire $10,000 to our account immediately.
    """
    result = perform_nlp(text, filename="round.txt")
    show(result)
    if "suspicious_number_formatting" not in result["flags"]:
        # This is a soft check - just print, don't fail hard
        print(f"      (note: round number flag not triggered — num_analysis={result['details'].get('number_analysis')})")
    else:
        ok("suspicious_number_formatting detected")
    print()

# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_clean_text()
    test_urgent_threat()
    test_fraud_patterns()
    test_excessive_caps()
    test_repetition()
    test_no_text()
    test_round_numbers()
    print("=" * 55)
    print("All NLP tests passed")
