import re
import io
import math
import pdfplumber
from PIL import Image

# Benford's Law expected frequencies for leading digits 1-9
BENFORD_EXPECTED = {
    1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 5: 0.079,
    6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
}

def extract_text(file_bytes: bytes, filename: str = "") -> str:
    """
    Extract text from a PDF or image file.
    - PDFs: uses pdfplumber
    - Images: returns empty string (no OCR for MVP)
    - Other: attempts UTF-8 decode
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                return "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except Exception as e:
            print(f"pdfplumber failed: {e}")
            return ""

    # Try to detect PDF by magic bytes even without extension
    if file_bytes[:4] == b"%PDF":
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                return "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except Exception as e:
            print(f"pdfplumber (magic) failed: {e}")
            return ""

    # Image fallback: no OCR in MVP
    try:
        Image.open(io.BytesIO(file_bytes))
        return ""  # image detected, no text to extract
    except Exception:
        pass

    # Plain text fallback
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_leading_digits(text: str) -> list[int]:
    """
    Find all numbers in the text and extract their leading digit.
    Filters: must be > 0, strips currency symbols and commas first.
    """
    # Normalize currency/thousands separators: $1,234.56 → 1234.56
    cleaned = re.sub(r"[$€£,]", "", text)
    # Find all numeric tokens (int or float)
    numbers_str = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned)
    leading_digits = []
    for n in numbers_str:
        # Strip leading zeros to get the true leading digit
        stripped = n.lstrip("0")
        if stripped and stripped[0].isdigit():
            d = int(stripped[0])
            if 1 <= d <= 9:
                leading_digits.append(d)
    return leading_digits


def chi_square_benford(digits: list[int]) -> float:
    """
    Compute a chi-square statistic between observed digit distribution
    and Benford's expected distribution.
    Returns the chi-square value (higher = more deviation from Benford's).
    """
    n = len(digits)
    if n == 0:
        return 0.0

    # Count observed occurrences
    observed = {d: 0 for d in range(1, 10)}
    for d in digits:
        observed[d] += 1

    chi2 = 0.0
    for d in range(1, 10):
        expected_count = BENFORD_EXPECTED[d] * n
        if expected_count > 0:
            chi2 += ((observed[d] - expected_count) ** 2) / expected_count

    return chi2


def perform_benfords(file_bytes: bytes, filename: str = "") -> dict:
    """
    Run Benford's Law analysis on the document.

    Returns a dict with:
      - score (0-100): 0 = perfectly Benford-conformant, 100 = highly suspicious
      - digit_count: number of leading digits analyzed
      - chi_square: raw chi-square statistic
      - message: human-readable explanation
    """
    text = extract_text(file_bytes, filename)

    if not text.strip():
        return {
            "score": 0,
            "digit_count": 0,
            "chi_square": 0.0,
            "message": "No text extracted — Benford's Law skipped (image or non-text file)."
        }

    digits = extract_leading_digits(text)

    if len(digits) < 10:
        return {
            "score": 0,
            "digit_count": len(digits),
            "chi_square": 0.0,
            "message": f"Too few numbers found ({len(digits)}) for a reliable Benford's analysis."
        }

    chi2 = chi_square_benford(digits)

    # Chi-square critical values (df=8):
    #   p=0.05 → 15.51  (not suspicious)
    #   p=0.01 → 20.09  (suspicious)
    #   p=0.001 → 26.12 (highly suspicious)
    # We map chi2 → 0-100 score using a soft cap at chi2=40 (extreme deviation)
    score = min(100, max(0, int((chi2 / 40.0) * 100)))

    if chi2 < 15.51:
        message = f"Distribution conforms to Benford's Law (χ²={chi2:.2f}). Low fraud signal."
    elif chi2 < 20.09:
        message = f"Mild deviation from Benford's Law (χ²={chi2:.2f}). Warrants closer inspection."
    elif chi2 < 26.12:
        message = f"Significant deviation from Benford's Law (χ²={chi2:.2f}). Numbers may be fabricated."
    else:
        message = f"Extreme deviation from Benford's Law (χ²={chi2:.2f}). High likelihood of financial fabrication."

    return {
        "score": score,
        "digit_count": len(digits),
        "chi_square": round(chi2, 4),
        "message": message
    }
