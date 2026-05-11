"""
Quick unit test for the Benford's Law layer.
Run with: .\\venv\\Scripts\\python.exe test_benfords.py
"""
import sys
from benfords import extract_leading_digits, perform_benfords

# --- Test 1: Benford-conformant distribution ---
# Roughly simulate text where leading digits follow Benford's Law
benford_text = (
    "Revenue: $1,234,567. Cost: $2,100. Profit: $345,000. "
    "Tax: $12,300. Assets: $1,000,000. Liabilities: $345,000. "
    "Expenses: $234,500. Income: $1,780. Interest: $567. "
    "Depreciation: $1,234. Net: $3,456. "
    "Q1: $145,000 Q2: $290,000 Q3: $178,000 Q4: $123,000 "
    "Invoice #1001 total $1,450.00 Invoice #1002 total $2,300.00 "
    "Invoice #1003 total $3,200.00 Invoice #1004 total $1,100.00 "
    "Salary: $120,000 Bonus: $15,000 Commission: $23,400 "
    "Equipment: $45,000 Software: $12,500 Rent: $36,000"
)

# --- Test 2: Suspicious (non-Benford) distribution ---
# Fabricated figures often cluster around "round" numbers or a single leading digit
suspicious_text = (
    "Payment: $5,000. Transfer: $5,500. Wire: $55,000. Invoice: $500. "
    "Fee: $5,100. Settlement: $50,000. Premium: $55,500. Deposit: $5,050. "
    "Refund: $5,200. Charge: $5,300. Balance: $55,100. Interest: $5,400. "
    "Principal: $5,600. Penalty: $5,700. Adjustment: $5,800. Credit: $5,900. "
    "Payment2: $50,100 Payment3: $50,200 Payment4: $50,300 Total: $500,000"
)

print("=" * 60)
print("TEST 1: Benford-conformant text")
print("=" * 60)
digits_1 = extract_leading_digits(benford_text)
print(f"  Digits found: {len(digits_1)}")
result_1 = perform_benfords(benford_text.encode(), filename="test.txt")
print(f"  Score:       {result_1['score']}/100")
print(f"  Chi-Square:  {result_1['chi_square']}")
print(f"  Message:     {result_1['message']}")

print()
print("=" * 60)
print("TEST 2: Suspicious (5-dominant) text")
print("=" * 60)
digits_2 = extract_leading_digits(suspicious_text)
print(f"  Digits found: {len(digits_2)}")
result_2 = perform_benfords(suspicious_text.encode(), filename="test.txt")
print(f"  Score:       {result_2['score']}/100")
print(f"  Chi-Square:  {result_2['chi_square']}")
print(f"  Message:     {result_2['message']}")

print()
if result_1["score"] < result_2["score"]:
    print("✅ PASS: Suspicious text scored higher than conformant text.")
else:
    print("❌ FAIL: Scoring logic needs review.")
    sys.exit(1)
