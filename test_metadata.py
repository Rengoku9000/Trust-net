"""
test_metadata.py — Unit tests for the Metadata extraction layer (Phase 4).
Run with:  .\\venv\\Scripts\\python.exe test_metadata.py
"""

import io
import sys

from metadata import perform_metadata


# ── helpers ───────────────────────────────────────────────────────────────────

def ok(label: str):
    print(f"  ✅  {label}")


def fail(label: str, detail: str):
    print(f"  ❌  {label}: {detail}")
    sys.exit(1)


def show(result: dict):
    print(f"      score={result['score']}  flags={result['flags']}")
    print(f"      msg  : {result['message']}")


# ── Test 1: plain-text file → no_metadata ────────────────────────────────────

def test_no_metadata():
    print("[Test 1] Plain-text file → expect no_metadata flag")
    content = b"Invoice total: $4,200.00\nDate: 2025-01-15"
    result  = perform_metadata(content, filename="invoice.txt")
    show(result)
    if "no_metadata" not in result["flags"]:
        fail("no_metadata", f"flags were {result['flags']}")
    ok("no_metadata flag present")
    print()


# ── Test 2: clean PNG (no EXIF) → low score ──────────────────────────────────

def test_clean_png():
    print("[Test 2] Clean PNG with no EXIF → expect no_metadata, low-ish score")
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(0, 128, 255)).save(buf, format="PNG")
    result = perform_metadata(buf.getvalue(), filename="clean.png")
    show(result)
    if result["score"] > 40:
        fail("clean PNG score", f"expected ≤ 40, got {result['score']}")
    ok("score within expected range")
    print()


# ── Test 3: JPEG with Photoshop EXIF → photo_editing_software flag ────────────

def test_photoshop_exif():
    print("[Test 3] JPEG with Photoshop in Software tag → expect photo_editing_software")
    from PIL import Image
    import piexif

    exif_dict = {
        "0th": {
            piexif.ImageIFD.Software: b"Adobe Photoshop CC 2023",
            piexif.ImageIFD.Make:     b"Canon",
            piexif.ImageIFD.Model:    b"EOS 5D",
        },
        "Exif": {},
        "GPS":  {},
        "1st":  {},
    }
    exif_bytes = piexif.dump(exif_dict)
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(255, 0, 0)).save(buf, format="JPEG", exif=exif_bytes)

    result = perform_metadata(buf.getvalue(), filename="tampered.jpg")
    show(result)
    if "photo_editing_software" not in result["flags"]:
        fail("photo_editing_software", f"flags were {result['flags']}")
    if result["score"] < 20:
        fail("score", f"expected ≥ 20, got {result['score']}")
    ok("photo_editing_software flag present, score elevated")
    print()


# ── Test 4: JPEG with GPS → gps_in_financial_doc flag ────────────────────────

def test_gps_exif():
    print("[Test 4] JPEG with GPS data → expect gps_in_financial_doc flag")
    from PIL import Image
    import piexif

    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef:  b"N",
        piexif.GPSIFD.GPSLatitude:     ((40, 1), (44, 1), (0, 1)),
        piexif.GPSIFD.GPSLongitudeRef: b"W",
        piexif.GPSIFD.GPSLongitude:    ((74, 1), (0, 1), (0, 1)),
    }
    exif_dict = {"0th": {}, "Exif": {}, "GPS": gps_ifd, "1st": {}}
    exif_bytes = piexif.dump(exif_dict)
    buf = io.BytesIO()
    Image.new("RGB", (100, 100)).save(buf, format="JPEG", exif=exif_bytes)

    result = perform_metadata(buf.getvalue(), filename="geo_tagged.jpg")
    show(result)
    if "gps_in_financial_doc" not in result["flags"]:
        fail("gps_in_financial_doc", f"flags were {result['flags']}")
    ok("gps_in_financial_doc flag present")
    print()


# ── Test 5: synthetic PDF-like bytes → graceful fallback ─────────────────────

def test_bad_pdf():
    print("[Test 5] Garbage bytes as .pdf → expect graceful no_metadata")
    content = b"%PDF-1.4 garbage content that cannot be parsed properly"
    result  = perform_metadata(content, filename="corrupt.pdf")
    show(result)
    # Should not crash; score may vary
    if not isinstance(result["score"], int):
        fail("graceful handling", "score is not an int")
    ok("handled gracefully, no crash")
    print()


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_no_metadata()
    test_clean_png()
    test_photoshop_exif()
    test_gps_exif()
    test_bad_pdf()
    print("=" * 50)
    print("All metadata tests passed ✅")
