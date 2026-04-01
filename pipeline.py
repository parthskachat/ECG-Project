"""
pipeline.py — Master Pipeline
==============================
Watches  data/pdfs/  for new Tricog ECG PDFs, digitises each one,
and writes the outputs that app.py (CardioScan Pro) expects:

    data/patients.json          ← patient registry  (list of records)
    data/csvs/<record_id>.csv   ← digitised mV data  (one file per PDF)

Usage
-----
Process a single PDF:
    python pipeline.py data/pdfs/r.pdf

Process all PDFs in data/pdfs/:
    python pipeline.py --all

Watch folder for new PDFs (continuous mode):
    python pipeline.py --watch
"""

import sys
import json
import time
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ── Path setup ─────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
CORE_DIR  = ROOT / "core"
DATA_DIR  = ROOT / "data"
PDF_DIR   = DATA_DIR / "pdfs"
CSV_DIR   = DATA_DIR / "csvs"
META_FILE = DATA_DIR / "patients.json"

sys.path.insert(0, str(CORE_DIR))

from digitizer        import digitize_pdf
from signal_processor import process_leads, compute_heart_rate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ═══════════════════════════════════════════════════════════════════════════
# Registry helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_registry() -> list:
    if META_FILE.exists():
        try:
            return json.loads(META_FILE.read_text())
        except json.JSONDecodeError:
            return []
    return []


def save_registry(records: list) -> None:
    META_FILE.write_text(json.dumps(records, indent=2))


def record_id_for(pdf_path: Path) -> str:
    """Stable short ID derived from filename + mtime."""
    raw = f"{pdf_path.name}-{pdf_path.stat().st_mtime}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]


def is_already_processed(record_id: str, registry: list) -> bool:
    return any(r.get("id") == record_id and r.get("status") == "ready"
               for r in registry)


# ═══════════════════════════════════════════════════════════════════════════
# PDF → patients.json + CSV
# ═══════════════════════════════════════════════════════════════════════════

def process_pdf(pdf_path: Path) -> dict:
    """
    Full pipeline for one PDF.  Returns a record dict suitable for patients.json.
    Raises on fatal errors so the caller can mark the record as 'error'.
    """
    log.info(f"Processing: {pdf_path.name}")
    record_id = record_id_for(pdf_path)

    # ── Mark as processing ────────────────────────────────────────────────
    registry = load_registry()
    # Remove any stale entry for this record
    registry = [r for r in registry if r.get("id") != record_id]
    stub = {
        "id":          record_id,
        "patient_name": pdf_path.stem,   # temporary name from filename
        "patient_id":  record_id[:6].upper(),
        "age":         "–",
        "gender":      "–",
        "test_date":   datetime.utcnow().isoformat() + "Z",
        "received_at": datetime.utcnow().isoformat() + "Z",
        "heart_rate":  0,
        "pr_interval": None,
        "qt_interval": None,
        "interpretation": "",
        "status":      "processing",
        "source_pdf":  pdf_path.name,
    }
    registry.append(stub)
    save_registry(registry)
    log.info(f"  Record {record_id} → status: processing")

    # ── Digitise ──────────────────────────────────────────────────────────
    raw_leads = digitize_pdf(pdf_path)
    fs        = float(raw_leads.get("__fs__", 295.0))

    # ── Clean signals ─────────────────────────────────────────────────────
    clean_leads = process_leads(raw_leads)

    # ── Compute HR from rhythm strip → Lead II → Lead I ──────────────────
    hr_bpm = 0
    for ref in ["II_rhythm", "II", "I"]:
        sig = clean_leads.get(ref)
        if sig is not None and len(sig) > 0:
            mhr, _, _ = compute_heart_rate(sig, fs)
            if mhr > 0:
                hr_bpm = int(round(mhr))
                break

    # ── Save CSV ──────────────────────────────────────────────────────────
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, sig in clean_leads.items():
        if name.startswith("__"):
            continue
        if sig is None or len(sig) == 0:
            continue
        for i, v in enumerate(sig):
            rows.append({
                "lead":   name,
                "sample": i,
                "time_s": round(i / fs, 5),
                "mv":     round(float(v), 6),
            })

    df = pd.DataFrame(rows)
    csv_path = CSV_DIR / f"{record_id}.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"  Saved CSV: {csv_path.name}  ({len(df)} rows, "
             f"{df['lead'].nunique()} leads)")

    # ── Build the finished record ─────────────────────────────────────────
    # Try to read PR/QT from the PDF text if pdfplumber is available
    pr_ms, qt_ms, interp_text, pat_name, age, gender = _extract_pdf_metadata(pdf_path)

    record = {
        "id":             record_id,
        "patient_name":   pat_name   or pdf_path.stem,
        "patient_id":     record_id[:6].upper(),
        "age":            age        or "–",
        "gender":         gender     or "–",
        "test_date":      datetime.utcnow().isoformat() + "Z",
        "received_at":    datetime.utcnow().isoformat() + "Z",
        "heart_rate":     hr_bpm,
        "pr_interval":    pr_ms,
        "qt_interval":    qt_ms,
        "interpretation": interp_text,
        "status":         "ready",
        "source_pdf":     pdf_path.name,
        "sample_rate_hz": round(fs, 1),
    }

    # ── Update registry ───────────────────────────────────────────────────
    registry = load_registry()
    registry = [r for r in registry if r.get("id") != record_id]
    registry.append(record)
    save_registry(registry)
    log.info(f"  Record {record_id} → status: ready  HR={hr_bpm} bpm")

    return record


def _extract_pdf_metadata(pdf_path: Path):
    """
    Try to read PR, QT, interpretation, patient name, age, gender from the
    Tricog PDF text layer using pdfplumber.
    Returns (pr_ms, qt_ms, interpretation, patient_name, age, gender).
    All values may be None if extraction fails.
    """
    pr_ms = qt_ms = interp_text = pat_name = age = gender = None
    try:
        import pdfplumber, re
        with pdfplumber.open(str(pdf_path)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)

        # PR interval  e.g. "PRI: 148ms"
        m = re.search(r"PR[I]?\s*:\s*(\d+)\s*ms", text, re.IGNORECASE)
        if m: pr_ms = int(m.group(1))

        # QT interval  e.g. "QT: 416ms"
        m = re.search(r"\bQT\s*:\s*(\d+)\s*ms", text, re.IGNORECASE)
        if m: qt_ms = int(m.group(1))

        # Patient name  — Tricog puts name as first bold line
        m = re.search(r"(?:Patient Name|Name)\s*[:\-]?\s*([A-Za-z ]{2,40})", text)
        if m: pat_name = m.group(1).strip()
        else:
            # Try first non-empty line after "Amritanshu Mishra"-style header
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if lines: pat_name = lines[0][:40]

        # Age / Gender  e.g. "51/Male"
        m = re.search(r"(\d{1,3})\s*/\s*(Male|Female|M|F)", text, re.IGNORECASE)
        if m:
            age    = int(m.group(1))
            gender = "M" if m.group(2).lower().startswith("m") else "F"

        # Interpretation — text between "Sinus" and "Disclaimer"
        m = re.search(
            r"(Sinus[\s\S]{10,600}?)(?:Disclaimer|REPORTED BY|$)",
            text, re.IGNORECASE
        )
        if m:
            interp_text = " ".join(m.group(1).split())

    except Exception as exc:
        log.debug(f"PDF metadata extraction skipped: {exc}")

    return pr_ms, qt_ms, interp_text, pat_name, age, gender


# ═══════════════════════════════════════════════════════════════════════════
# CLI modes
# ═══════════════════════════════════════════════════════════════════════════

def run_single(pdf_path: Path) -> None:
    if not pdf_path.exists():
        log.error(f"File not found: {pdf_path}")
        sys.exit(1)
    try:
        rec = process_pdf(pdf_path)
        log.info(f"Done: {rec['patient_name']}  HR={rec['heart_rate']} bpm  "
                 f"status={rec['status']}")
    except Exception as exc:
        log.error(f"Pipeline failed for {pdf_path.name}: {exc}", exc_info=True)
        _mark_error(pdf_path, str(exc))


def run_all() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        log.warning(f"No PDFs found in {PDF_DIR}")
        return
    log.info(f"Found {len(pdfs)} PDF(s)")
    for pdf in pdfs:
        try:
            process_pdf(pdf)
        except Exception as exc:
            log.error(f"Failed: {pdf.name}  —  {exc}", exc_info=True)
            _mark_error(pdf, str(exc))


def run_watch(poll_sec: float = 5.0) -> None:
    """Poll PDF_DIR every poll_sec seconds and process new files."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Watching {PDF_DIR}  (polling every {poll_sec}s)  Ctrl-C to stop")
    seen = set()

    while True:
        try:
            for pdf in sorted(PDF_DIR.glob("*.pdf")):
                registry = load_registry()
                rid = record_id_for(pdf)
                if rid not in seen and not is_already_processed(rid, registry):
                    seen.add(rid)
                    try:
                        process_pdf(pdf)
                    except Exception as exc:
                        log.error(f"Failed: {pdf.name}: {exc}", exc_info=True)
                        _mark_error(pdf, str(exc))
            time.sleep(poll_sec)
        except KeyboardInterrupt:
            log.info("Watcher stopped.")
            break


def _mark_error(pdf_path: Path, msg: str) -> None:
    rid = record_id_for(pdf_path)
    registry = load_registry()
    for r in registry:
        if r.get("id") == rid:
            r["status"] = "error"
            r["error_message"] = msg[:200]
    save_registry(registry)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tricog ECG Digitizer Pipeline"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("pdf",     nargs="?", type=Path,
                       help="Path to a single PDF to process")
    group.add_argument("--all",   action="store_true",
                       help="Process all PDFs in data/pdfs/")
    group.add_argument("--watch", action="store_true",
                       help="Watch data/pdfs/ for new files continuously")
    parser.add_argument("--poll", type=float, default=5.0,
                        help="Poll interval in seconds for --watch (default: 5)")

    args = parser.parse_args()

    # Create required directories
    for d in [DATA_DIR, PDF_DIR, CSV_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if args.watch:
        run_watch(poll_sec=args.poll)
    elif args.all:
        run_all()
    elif args.pdf:
        run_single(args.pdf)
    else:
        parser.print_help()
