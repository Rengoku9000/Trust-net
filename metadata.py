"""
metadata.py — TrustNet Phase 4: Metadata Extraction & Anomaly Scoring
-----------------------------------------------------------------------
Analyses document metadata for tampering indicators:
  - PDF: author, creator, producer, creation/modification dates
  - Images: EXIF software, camera info, GPS, timestamps

Score 0-100. Higher = more suspicious.
"""

import io
import re
from datetime import datetime
from typing import Optional, Dict, Any, List

# ── Suspicious software signatures ───────────────────────────────────────────

PHOTO_EDITORS = [
    "photoshop", "gimp", "lightroom", "affinity photo", "pixelmator",
    "paint.net", "photoscape", "canva", "snapseed", "picsart", "picasa",
    "luminar", "darktable", "rawtherapee",
]

SUSPICIOUS_PDF_TOOLS = [
    "ilovepdf", "smallpdf", "pdf24", "sejda", "pdfescape",
    "pdfzorro", "pdf candy", "sodapdf", "foxit phantom", "pdfsam",
]

# ── Flag weights (points added to risk score per flag) ────────────────────────

FLAG_WEIGHTS: Dict[str, int] = {
    "no_metadata":              35,   # No metadata at all — unusual for legit docs
    "mod_before_creation":      50,   # Impossible timestamp relationship
    "future_date":              30,   # Creation date is in the future
    "photo_editing_software":   25,   # Photoshop/GIMP in a financial document
    "suspicious_pdf_tool":      20,   # Known forgery-enabling tool as producer
    "modified_long_after":      15,   # Modified years after creation
    "missing_author":           10,   # No author field
    "gps_in_financial_doc":     10,   # GPS coords in an image doc
}

# ── PDF date parsing ─────────────────────────────────────────────────────────

def _parse_pdf_date(date_str: str) -> Optional[datetime]:
    """Parse PDF date format: D:YYYYMMDDHHmmSSZ"""
    if not date_str:
        return None
    s = date_str.strip()
    if s.startswith("D:"):
        s = s[2:]
    # Strip timezone designator (e.g. +05'30' or Z)
    s = re.sub(r"[Z+\-]\d{2}'?\d{2}'?$", "", s)
    s = s[:14]  # Keep at most YYYYMMDDHHmmSS
    for fmt in ["%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d"]:
        try:
            return datetime.strptime(s[: len(fmt.replace("%Y", "0000").replace("%m", "00")
                                          .replace("%d", "00").replace("%H", "00")
                                          .replace("%M", "00").replace("%S", "00"))], fmt)
        except ValueError:
            continue
    return None

# ── PDF analysis ─────────────────────────────────────────────────────────────

def _analyze_pdf(content: bytes) -> Dict[str, Any]:
    """Extract and score metadata from a PDF file."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content))
        meta = reader.metadata or {}
    except Exception as exc:
        return {"flags": ["no_metadata"], "details": {"parse_error": str(exc)}}

    flags: List[str] = []
    details: Dict[str, Any] = {}

    author   = str(meta.get("/Author",       "") or "").strip()
    creator  = str(meta.get("/Creator",      "") or "").strip()
    producer = str(meta.get("/Producer",     "") or "").strip()
    cdate    = str(meta.get("/CreationDate", "") or "").strip()
    mdate    = str(meta.get("/ModDate",      "") or "").strip()

    details.update({
        "author":            author   or None,
        "creator":           creator  or None,
        "producer":          producer or None,
        "creation_date_raw": cdate    or None,
        "mod_date_raw":      mdate    or None,
    })

    # No metadata at all
    if not any([author, creator, producer, cdate, mdate]):
        flags.append("no_metadata")
        return {"flags": flags, "details": details}

    # Missing author
    if not author:
        flags.append("missing_author")

    # Photo editing software / suspicious PDF tool
    combo = (creator + " " + producer).lower()
    for ed in PHOTO_EDITORS:
        if ed in combo:
            flags.append("photo_editing_software")
            details["editing_software"] = ed
            break
    for tool in SUSPICIOUS_PDF_TOOLS:
        if tool in combo:
            flags.append("suspicious_pdf_tool")
            details["suspicious_tool"] = tool
            break

    # Date anomaly analysis
    creation_dt = _parse_pdf_date(cdate)
    mod_dt      = _parse_pdf_date(mdate)
    now         = datetime.utcnow()

    if creation_dt:
        details["creation_date"] = creation_dt.isoformat()
        if creation_dt > now:
            flags.append("future_date")
            details["future_date_delta_days"] = (creation_dt - now).days

    if creation_dt and mod_dt:
        details["mod_date"] = mod_dt.isoformat()
        delta_secs = (mod_dt - creation_dt).total_seconds()

        if delta_secs < -60:                           # 1-min tolerance
            flags.append("mod_before_creation")
            details["mod_before_creation_by_seconds"] = abs(int(delta_secs))
        elif delta_secs > 365 * 24 * 3600:             # Modified 1+ year later
            flags.append("modified_long_after")
            details["modified_days_after_creation"] = int(delta_secs / 86400)

    return {"flags": flags, "details": details}

# ── Image / EXIF analysis ────────────────────────────────────────────────────

def _analyze_image(content: bytes) -> Dict[str, Any]:
    """Extract and score EXIF metadata from an image file."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        img  = Image.open(io.BytesIO(content))
        raw  = img._getexif()  # type: ignore[attr-defined]
    except Exception as exc:
        return {"flags": ["no_metadata"], "details": {"parse_error": str(exc)}}

    if not raw:
        return {"flags": ["no_metadata"], "details": {"reason": "No EXIF data present"}}

    flags: List[str] = []
    details: Dict[str, Any] = {}

    exif = {TAGS.get(k, k): v for k, v in raw.items()}

    software = str(exif.get("Software", "") or "").strip()
    make     = str(exif.get("Make",     "") or "").strip()
    model    = str(exif.get("Model",    "") or "").strip()
    dt_orig  = exif.get("DateTimeOriginal")
    gps_info = exif.get("GPSInfo")

    details.update({
        "software":        software or None,
        "camera_make":     make     or None,
        "camera_model":    model    or None,
        "datetime_original": str(dt_orig) if dt_orig else None,
        "has_gps":         bool(gps_info),
    })

    # Photo editing software
    if software:
        for ed in PHOTO_EDITORS:
            if ed in software.lower():
                flags.append("photo_editing_software")
                details["editing_software"] = software
                break

    # GPS in what is supposed to be a financial document
    if gps_info:
        flags.append("gps_in_financial_doc")

    # Future timestamp
    if dt_orig:
        try:
            dt = datetime.strptime(str(dt_orig), "%Y:%m:%d %H:%M:%S")
            if dt > datetime.utcnow():
                flags.append("future_date")
                details["future_date_delta_days"] = (dt - datetime.utcnow()).days
        except ValueError:
            pass

    return {"flags": flags, "details": details}

# ── Public API ────────────────────────────────────────────────────────────────

def perform_metadata(content: bytes, filename: str = "") -> Dict[str, Any]:
    """
    Main entry point.  Detect file type, run appropriate metadata analysis,
    and return a normalised score (0–100) with flags and detail.
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        result = _analyze_pdf(content)
    elif ext in {"jpg", "jpeg", "png", "tiff", "tif", "bmp", "webp"}:
        result = _analyze_image(content)
    else:
        # Unknown extension: try PDF first, fall back to image
        result = _analyze_pdf(content)
        if result.get("details", {}).get("parse_error"):
            result = _analyze_image(content)

    triggered = result.get("flags", [])
    score      = min(100, sum(FLAG_WEIGHTS.get(f, 0) for f in triggered))

    if score >= 70:
        message = "HIGH: Severe metadata anomalies — strong indicators of tampering."
    elif score >= 35:
        message = "MEDIUM: Metadata irregularities detected — warrants closer inspection."
    else:
        message = "LOW: Metadata appears consistent with a legitimate document."

    return {
        "score":   score,
        "flags":   triggered,
        "message": message,
        "details": result.get("details", {}),
    }
