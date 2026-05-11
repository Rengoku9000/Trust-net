"""
test_gnn.py — Unit tests for the Graph Network Analysis layer (Phase 6).
Run with:  .\\venv\\Scripts\\python.exe test_gnn.py
"""

import sys
from gnn import perform_gnn

# ── helpers ───────────────────────────────────────────────────────────────────

def ok(label: str):
    print(f"  [PASS]  {label}")

def fail(label: str, detail: str):
    print(f"  [FAIL]  {label}: {detail}")
    sys.exit(1)

def show(result: dict):
    print(f"      score={result['score']}  flags={result['flags']}")
    print(f"      msg  : {result['message']}")
    d = result["details"]
    for k in ("amounts_found", "transaction_pairs_found", "graph_nodes", "graph_edges"):
        if k in d:
            print(f"      {k}: {d[k]}")

# ── Test 1: clean document -> low score ──────────────────────────────────────

def test_clean():
    print("[Test 1] Clean invoice -> expect low score")
    text = b"""
    Invoice #INV-2025-001   Date: 2025-01-15
    Consulting services: $6,600.00
    Tax:                   $660.00
    Total:               $7,260.00

    Payment due: 2025-02-15
    """
    result = perform_gnn(text, filename="invoice.txt")
    show(result)
    if result["score"] > 35:
        fail("clean score", f"expected <= 35, got {result['score']}")
    ok(f"low score {result['score']}")
    print()

# ── Test 2: structuring / smurfing ────────────────────────────────────────────

def test_structuring():
    print("[Test 2] Amounts just below $10,000 -> expect structuring_detected")
    text = b"""
    Transaction Log - Week of 2025-03-10

    Ref WIRE-001  Amount: $9,800   Date: 2025-03-10
    Ref WIRE-002  Amount: $9,750   Date: 2025-03-11
    Ref WIRE-003  Amount: $9,900   Date: 2025-03-12
    Ref WIRE-004  Amount: $9,500   Date: 2025-03-13

    All transfers sent to offshore account ACCT-77812345.
    """
    result = perform_gnn(text, filename="transfers.txt")
    show(result)
    if "structuring_detected" not in result["flags"]:
        fail("structuring_detected", f"flags={result['flags']}, detail={result['details']}")
    ok(f"structuring_detected  score={result['score']}")
    print()

# ── Test 3: duplicate amounts (template fraud) ────────────────────────────────

def test_duplicate_amounts():
    print("[Test 3] Same amount repeated 5x -> expect duplicate_amounts")
    text = b"""
    Invoice batch:
    INV-001: $4,500.00  due 2025-01-10
    INV-002: $4,500.00  due 2025-01-11
    INV-003: $4,500.00  due 2025-01-12
    INV-004: $4,500.00  due 2025-01-13
    INV-005: $4,500.00  due 2025-01-14
    """
    result = perform_gnn(text, filename="batch.txt")
    show(result)
    if "duplicate_amounts" not in result["flags"]:
        fail("duplicate_amounts", f"flags={result['flags']}, detail={result['details']}")
    ok(f"duplicate_amounts detected  score={result['score']}")
    print()

# ── Test 4: circular flow A->B->C->A ─────────────────────────────────────────

def test_circular_flow():
    print("[Test 4] Circular A->B->C->A transfers -> expect circular_flow")
    text = b"""
    Wire Transfer Audit Log

    From: ACCT-1111  To: ACCT-2222   Amount: $50,000  Date: 2025-04-01
    From: ACCT-2222  To: ACCT-3333   Amount: $48,500  Date: 2025-04-02
    From: ACCT-3333  To: ACCT-1111   Amount: $47,000  Date: 2025-04-03
    """
    result = perform_gnn(text, filename="circular.txt")
    show(result)
    if "circular_flow" not in result["flags"]:
        fail("circular_flow", f"flags={result['flags']}, detail={result['details']}")
    ok(f"circular_flow detected  score={result['score']}")
    print()

# ── Test 5: hub-and-spoke (money mule) ───────────────────────────────────────

def test_hub_spoke():
    print("[Test 5] One sender to 5 distinct receivers -> expect hub_and_spoke")
    text = b"""
    Outgoing Transfers from ACCT-9999:

    From: ACCT-9999  To: ACCT-1001   Amount: $8,000   Date: 2025-05-01
    From: ACCT-9999  To: ACCT-1002   Amount: $7,500   Date: 2025-05-01
    From: ACCT-9999  To: ACCT-1003   Amount: $9,200   Date: 2025-05-01
    From: ACCT-9999  To: ACCT-1004   Amount: $6,800   Date: 2025-05-01
    From: ACCT-9999  To: ACCT-1005   Amount: $8,100   Date: 2025-05-01
    """
    result = perform_gnn(text, filename="hub.txt")
    show(result)
    if "hub_and_spoke" not in result["flags"]:
        fail("hub_and_spoke", f"flags={result['flags']}, detail={result['details']}")
    ok(f"hub_and_spoke detected  score={result['score']}")
    print()

# ── Test 6: self-dealing ──────────────────────────────────────────────────────

def test_self_dealing():
    print("[Test 6] Sender == Receiver -> expect self_dealing")
    text = b"""
    Internal Transfer Record

    From: ACCT-5555  To: ACCT-5555   Amount: $25,000  Date: 2025-06-01
    From: ACCT-5555  To: ACCT-5555   Amount: $12,000  Date: 2025-06-02
    """
    result = perform_gnn(text, filename="self_deal.txt")
    show(result)
    if "self_dealing" not in result["flags"]:
        fail("self_dealing", f"flags={result['flags']}, detail={result['details']}")
    ok(f"self_dealing detected  score={result['score']}")
    print()

# ── Test 7: binary/empty content -> graceful fallback ─────────────────────────

def test_no_entities():
    print("[Test 7] Binary content -> expect no_entities_found, no crash")
    result = perform_gnn(bytes([0xFF, 0xD8, 0xFF, 0xE0]), filename="photo.jpg")
    show(result)
    if not isinstance(result["score"], int):
        fail("graceful handling", "score is not int")
    ok("handled gracefully")
    print()

# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_clean()
    test_structuring()
    test_duplicate_amounts()
    test_circular_flow()
    test_hub_spoke()
    test_self_dealing()
    test_no_entities()
    print("=" * 55)
    print("All GNN tests passed")
