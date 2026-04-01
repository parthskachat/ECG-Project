#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${HOME}/ecg_project"
cd "$PROJECT_DIR"

echo "========================================"
echo "  CardioScan Pro - Setup and Run"
echo "========================================"

# Create virtualenv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo ""
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo ""
echo "Ensuring directory structure..."
mkdir -p data/pdfs data/csvs core dashboard

# Verify r.pdf exists
if [ ! -f "data/pdfs/r.pdf" ]; then
    echo "WARNING: data/pdfs/r.pdf not found!"
    echo "  Place your Tricog ECG PDF at: ${PROJECT_DIR}/data/pdfs/r.pdf"
    echo "  Then re-run this script."
fi

echo ""
echo "========================================"
echo "  Running Pipeline"
echo "========================================"
echo ""

# Clear previous processing results to force re-processing
# (remove the specific record from patients.json if it exists)
if [ -f "data/patients.json" ]; then
    echo "Clearing previous results to force reprocessing..."
    echo "[]" > data/patients.json
    rm -f data/csvs/*.csv
fi

python3 run_pipeline.py

echo ""
echo "========================================"
echo "  Validation"
echo "========================================"
echo ""

python3 -c "
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as scipy_signal

csv_dir = Path('data/csvs')
csv_files = sorted(csv_dir.glob('*.csv'), key=lambda f: f.stat().st_mtime, reverse=True)

if not csv_files:
    print('ERROR: No CSV files found')
    exit(1)

df = pd.read_csv(csv_files[0])
print(f'CSV: {csv_files[0].name}')
print(f'Total rows: {len(df)}')
print()

print(f'{\"Lead\":<12} {\"p-p mV\":>8} {\"HR bpm\":>8}')
print('-' * 32)

for lead_name in ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6','II_rhythm']:
    grp = df[df['lead'] == lead_name]
    if len(grp) == 0:
        print(f'{lead_name:<12} {\"MISSING\":>8} {\"--\":>8}')
        continue
    sig = grp['mv'].values
    pp = float(np.ptp(sig))

    # Compute HR with Pan-Tompkins
    fs = 1.0 / grp['time_s'].diff().median() if len(grp) > 1 else 299.0
    hr_str = '--'
    if len(sig) > int(0.6 * fs):
        try:
            nyq = fs / 2.0
            lo = min(5.0 / nyq, 0.99)
            hi = min(15.0 / nyq, 0.99)
            if lo < hi:
                b, a = scipy_signal.butter(2, [lo, hi], btype='band')
                filt = scipy_signal.filtfilt(b, a, sig)
                d = np.diff(filt)
                d = np.append(d, 0)
                sq = d ** 2
                wl = max(int(0.15 * fs), 3)
                intg = np.convolve(sq, np.ones(wl)/wl, mode='same')
                th = 0.5 * np.max(intg)
                md = max(int(0.4 * fs), 1)
                pks, _ = scipy_signal.find_peaks(intg, height=th*0.3, distance=md, prominence=th*0.2)
                if len(pks) >= 2:
                    rr = np.diff(pks) / fs
                    hrs = 60.0 / rr
                    hrs = hrs[(hrs >= 30) & (hrs <= 200)]
                    if len(hrs) > 0:
                        hr_str = f'{np.mean(hrs):.1f}'
        except:
            pass

    print(f'{lead_name:<12} {pp:>8.3f} {hr_str:>8}')

print()
print('VALIDATION:')
v4 = df[df['lead']=='V4']['mv'].values
v5 = df[df['lead']=='V5']['mv'].values
print(f'  V4 p-p >= 2.4 mV: {np.ptp(v4):.3f} mV  [{\"PASS\" if np.ptp(v4) >= 2.4 else \"CHECK\"}]')
print(f'  V5 p-p >= 2.3 mV: {np.ptp(v5):.3f} mV  [{\"PASS\" if np.ptp(v5) >= 2.3 else \"CHECK\"}]')

meta = json.loads(Path('data/patients.json').read_text())
if meta:
    hr = meta[-1].get('heart_rate', 0)
    sr = meta[-1].get('sample_rate_hz', 0)
    print(f'  Heart rate: {hr} bpm  [{\"PASS\" if hr and 55 <= hr <= 65 else \"CHECK\"}]')
    print(f'  Sample rate: {sr} Hz  [{\"PASS\" if sr and 280 < sr < 320 else \"CHECK\"}]')
"

echo ""
echo "========================================"
echo "  Launching Dashboard"
echo "========================================"
echo ""
echo "  Open in browser: http://localhost:8501"
echo "  Press Ctrl+C to stop."
echo ""

streamlit run dashboard/app.py --server.headless=true
