"""
run_pipeline.py — Scan data/pdfs/, digitize, process, write CSV + JSON.
"""

import json
import logging
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.digitizer import digitize_pdf
from core.signal_processor import process_leads, compute_heart_rate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ecg.pipeline")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
PDF_DIR   = DATA_DIR / "pdfs"
CSV_DIR   = DATA_DIR / "csvs"
META_FILE = DATA_DIR / "patients.json"

CSV_DIR.mkdir(parents=True, exist_ok=True)


def file_hash(path: Path) -> str:
    """Quick MD5 hash of a file for dedup."""
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()[:12]


def load_registry() -> list:
    if META_FILE.exists():
        return json.loads(META_FILE.read_text())
    return []


def save_registry(registry: list):
    META_FILE.write_text(json.dumps(registry, indent=2, default=str))


def process_one_pdf(pdf_path: Path) -> dict:
    """Full pipeline for a single PDF. Returns patient record dict."""
    log.info(f"{'='*60}")
    log.info(f"Processing: {pdf_path.name}")
    log.info(f"{'='*60}")

    record_id = file_hash(pdf_path)

    # ── Step 1: Digitize ──────────────────────────────────────────────────
    raw = digitize_pdf(pdf_path)
    fs  = float(raw.pop("__fs__", 299.0))
    log.info(f"Digitized {len(raw)} leads at {fs:.1f} Hz")

    # ── Step 2: Process (clean) ───────────────────────────────────────────
    raw["__fs__"] = fs
    cleaned = process_leads(raw, fs=fs)
    cleaned.pop("__fs__", None)
    raw.pop("__fs__", None)

    # ── Step 3: Compute metrics ───────────────────────────────────────────
    # Use Lead II for heart rate (standard clinical practice)
    hr_lead = cleaned.get("II", cleaned.get("II_rhythm", None))
    if hr_lead is not None:
        mean_hr, min_hr, max_hr = compute_heart_rate(hr_lead, fs)
    else:
        mean_hr, min_hr, max_hr = 0, 0, 0

    log.info(f"Heart rate: {mean_hr} bpm (range {min_hr}-{max_hr})")

    # ── Step 4: Write CSV ─────────────────────────────────────────────────
    rows = []
    for lead_name, sig in cleaned.items():
        for i, mv in enumerate(sig):
            rows.append({
                "lead":   lead_name,
                "sample": i,
                "time_s": round(i / fs, 6),
                "mv":     round(float(mv), 6),
            })

    df = pd.DataFrame(rows)
    csv_path = CSV_DIR / f"{record_id}.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"CSV written: {csv_path.name}  ({len(df)} rows)")

    # ── Step 5: Build patient record ──────────────────────────────────────
    record = {
        "id":             record_id,
        "patient_id":     pdf_path.stem,
        "patient_name":   pdf_path.stem.replace("_", " ").title(),
        "age":            None,
        "gender":         None,
        "test_date":      datetime.now().isoformat(),
        "received_at":    datetime.now().isoformat(),
        "heart_rate":     int(round(mean_hr)) if mean_hr > 0 else None,
        "pr_interval":    None,
        "qt_interval":    None,
        "sample_rate_hz": round(fs, 1),
        "interpretation": "",
        "status":         "ready",
        "source_pdf":     pdf_path.name,
    }

    # ── Print summary ─────────────────────────────────────────────────────
    log.info(f"")
    log.info(f"{'Lead':<12} {'p-p mV':>8} {'min':>8} {'max':>8} {'samples':>8}")
    log.info(f"{'-'*48}")
    for lead_name in ["I","II","III","aVR","aVL","aVF",
                       "V1","V2","V3","V4","V5","V6","II_rhythm"]:
        if lead_name in cleaned:
            sig = cleaned[lead_name]
            log.info(f"{lead_name:<12} {np.ptp(sig):>8.3f} {sig.min():>8.3f} "
                     f"{sig.max():>8.3f} {len(sig):>8}")

    return record


def main():
    log.info("CardioScan Pro Pipeline")
    log.info(f"Scanning: {PDF_DIR}")

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        log.warning(f"No PDF files found in {PDF_DIR}")
        return

    log.info(f"Found {len(pdfs)} PDF(s)")

    registry = load_registry()
    existing_ids = {r.get("id") for r in registry}

    for pdf_path in pdfs:
        rid = file_hash(pdf_path)
        if rid in existing_ids:
            log.info(f"Skipping {pdf_path.name} (already processed: {rid})")
            continue

        try:
            record = process_one_pdf(pdf_path)
            registry.append(record)
            save_registry(registry)
            log.info(f"Record saved: {record['id']}")
        except Exception as exc:
            log.error(f"Failed to process {pdf_path.name}: {exc}", exc_info=True)
            registry.append({
                "id":           file_hash(pdf_path),
                "patient_id":   pdf_path.stem,
                "patient_name": pdf_path.stem.replace("_", " ").title(),
                "status":       "error",
                "error":        str(exc),
                "source_pdf":   pdf_path.name,
                "test_date":    datetime.now().isoformat(),
            })
            save_registry(registry)

    log.info(f"Pipeline complete. {len(registry)} total records.")


if __name__ == "__main__":
    main()
