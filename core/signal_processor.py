"""
signal_processor.py — Morphology-Preserving ECG Signal Processor
=================================================================
Takes raw mV arrays from digitizer.py and applies a clinical-grade
DSP pipeline that cleans noise WITHOUT blunting QRS complexes.

Pipeline (per lead)
-------------------
1. 50 Hz IIR notch   — removes powerline interference (India standard)
2. 0.5 Hz high-pass  — removes baseline wander (AHA standard)
3. Cubic spline BC   — removes residual respiratory drift (polarity-aware)
4. 100 Hz low-pass   — removes high-frequency noise; preserves QRS harmonics
5. Mean-centering     — simulates clinical AC-coupling

Usage
-----
    from signal_processor import process_leads
    cleaned = process_leads(raw_leads_dict, fs=299.2)
"""

import logging
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from scipy import signal
from scipy.interpolate import UnivariateSpline

warnings.filterwarnings("ignore")
log = logging.getLogger("ecg.processor")

# ── Filter parameters ─────────────────────────────────────────────────────────
FS_DEFAULT        = 295.0
NOTCH_FREQ        = 50.0
NOTCH_Q           = 30.0
HP_CUTOFF         = 0.5
HP_ORDER          = 4
LP_CUTOFF         = 100.0
LP_ORDER          = 2
SPLINE_SMOOTH     = 0.50
SPLINE_WIN_SEC    = 0.20


# ═══════════════════════════════════════════════════════════════════════════════
# Individual filter stages
# ═══════════════════════════════════════════════════════════════════════════════

def apply_notch(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Remove 50 Hz powerline interference."""
    nyq = fs / 2.0
    if NOTCH_FREQ >= nyq:
        return ecg
    b, a = signal.iirnotch(NOTCH_FREQ / nyq, NOTCH_Q)
    return signal.filtfilt(b, a, ecg)


def apply_highpass(ecg: np.ndarray, fs: float) -> np.ndarray:
    """High-pass Butterworth at 0.5 Hz to remove gross baseline drift."""
    nyq  = fs / 2.0
    norm = min(HP_CUTOFF / nyq, 0.999)
    b, a = signal.butter(HP_ORDER, norm, btype="high")
    return signal.filtfilt(b, a, ecg)


def apply_spline_baseline(ecg: np.ndarray, fs: float) -> np.ndarray:
    """
    Remove residual respiratory wander via polarity-aware cubic spline.

    - Positive leads (mean > +0.15 mV): spline through local MINIMA
    - Negative leads (mean < -0.15 mV): spline through local MAXIMA
    - Mixed leads (|mean| <= 0.15 mV): spline through average of min and max
    """
    n   = len(ecg)
    win = max(int(SPLINE_WIN_SEC * fs), 10)
    step = max(win // 2, 1)

    # Determine lead polarity from signal mean
    sig_mean = np.mean(ecg)

    if sig_mean < -0.15:
        polarity = "negative"
    elif sig_mean > 0.15:
        polarity = "positive"
    else:
        polarity = "mixed"

    def _collect_extrema(ecg_arr, mode="min"):
        """Collect local min or max positions in sliding windows."""
        xi, yi = [], []
        for start in range(0, n - win, step):
            chunk = ecg_arr[start:start + win]
            if mode == "min":
                idx = int(np.argmin(chunk)) + start
            else:
                idx = int(np.argmax(chunk)) + start
            xi.append(idx)
            yi.append(ecg_arr[idx])
        return xi, yi

    if polarity == "positive":
        xi, yi = _collect_extrema(ecg, mode="min")
    elif polarity == "negative":
        xi, yi = _collect_extrema(ecg, mode="max")
    else:
        # Mixed: average of local min and max
        xi_min, yi_min = _collect_extrema(ecg, mode="min")
        xi_max, yi_max = _collect_extrema(ecg, mode="max")
        xi = []
        yi = []
        for i in range(min(len(xi_min), len(xi_max))):
            avg_x = (xi_min[i] + xi_max[i]) // 2
            avg_y = (yi_min[i] + yi_max[i]) / 2.0
            xi.append(avg_x)
            yi.append(avg_y)

    if len(xi) < 4:
        return ecg

    xi_arr = np.array(xi, dtype=float)
    yi_arr = np.array(yi, dtype=float)
    _, unique_idx = np.unique(xi_arr, return_index=True)
    xi_arr = xi_arr[unique_idx]
    yi_arr = yi_arr[unique_idx]

    if len(xi_arr) < 4:
        return ecg

    # Sort by x
    sort_idx = np.argsort(xi_arr)
    xi_arr = xi_arr[sort_idx]
    yi_arr = yi_arr[sort_idx]

    try:
        sp       = UnivariateSpline(xi_arr, yi_arr,
                                    s=len(xi_arr) * SPLINE_SMOOTH, k=3, ext=3)
        baseline = sp(np.arange(n))
        baseline = np.clip(baseline, -5.0, 5.0)
        return ecg - baseline
    except Exception as exc:
        log.debug(f"Spline baseline failed: {exc}")
        return ecg


def apply_lowpass(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Morphology-preserving low-pass at 100 Hz, order 2."""
    nyq  = fs / 2.0
    freq = min(LP_CUTOFF, nyq * 0.95)
    b, a = signal.butter(LP_ORDER, freq / nyq, btype="low")
    return signal.filtfilt(b, a, ecg)


def mean_center(ecg: np.ndarray) -> np.ndarray:
    """Subtract signal mean (simulates clinical AC-coupling)."""
    return ecg - ecg.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Complete single-lead pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def clean_lead(ecg: np.ndarray,
               fs:  float = FS_DEFAULT) -> np.ndarray:
    """Full morphology-preserving cleaning pipeline for a single lead."""
    if len(ecg) < 20:
        return ecg

    n   = len(ecg)
    out = ecg.copy()

    out = apply_notch(out,          fs)
    out = apply_highpass(out,       fs)
    out = apply_spline_baseline(out, fs)
    out = apply_lowpass(out,        fs)
    out = mean_center(out)

    if len(out) != n:
        out = out[:n] if len(out) > n else np.pad(out, (0, n - len(out)), "edge")

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-lead batch processor
# ═══════════════════════════════════════════════════════════════════════════════

def process_leads(raw_leads: Dict[str, np.ndarray],
                  fs: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Apply clean_lead() to every lead in a dictionary."""
    if fs is None:
        fs = float(raw_leads.get("__fs__", FS_DEFAULT))
    log.info(f"Processing {len(raw_leads)-1} leads at {fs:.1f} Hz")

    cleaned: Dict[str, np.ndarray] = {}
    for name, arr in raw_leads.items():
        if name == "__fs__":
            cleaned[name] = arr
            continue
        if arr is None or len(arr) == 0:
            log.warning(f"[{name}] Empty signal -- skipped")
            continue
        try:
            cleaned[name] = clean_lead(arr, fs)
            log.debug(f"[{name}] cleaned  p-p={np.ptp(cleaned[name]):.3f} mV")
        except Exception as exc:
            log.error(f"[{name}] Processing failed ({exc}) -- returning raw")
            cleaned[name] = arr

    return cleaned


# ═══════════════════════════════════════════════════════════════════════════════
# R-peak detection & HR calculation (Pan-Tompkins)
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_r_peaks_fallback(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Simple fallback R-peak detector when Pan-Tompkins fails."""
    if len(ecg) < int(0.5 * fs):
        return np.array([], dtype=int)
    try:
        abs_sig = np.abs(ecg - np.median(ecg))
        ht = 0.40 * np.ptp(abs_sig)
        dist = max(int(0.50 * fs), 1)
        peaks, _ = signal.find_peaks(abs_sig, height=ht, distance=dist)
        return peaks
    except Exception:
        return np.array([], dtype=int)


def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """
    Pan-Tompkins-inspired R-peak detector for clinical-grade HR calculation.

    Steps:
    1. Bandpass 5-15 Hz to isolate QRS energy (rejects P/T waves)
    2. Differentiate to emphasise steep QRS slopes
    3. Square to make all values positive and amplify large slopes
    4. Moving window integration (150 ms window)
    5. Adaptive thresholding with refractory period
    6. Refine: find true peak in original signal within +/-50 ms
    """
    if len(ecg) < int(0.6 * fs):
        return np.array([], dtype=int)

    try:
        nyq = fs / 2.0

        # Step 1: Bandpass 5-15 Hz
        low_cut = min(5.0 / nyq, 0.99)
        high_cut = min(15.0 / nyq, 0.99)
        if low_cut >= high_cut:
            return _detect_r_peaks_fallback(ecg, fs)
        b_bp, a_bp = signal.butter(2, [low_cut, high_cut], btype="band")
        filtered = signal.filtfilt(b_bp, a_bp, ecg)

        # Step 2: Differentiate
        diff_sig = np.diff(filtered)
        diff_sig = np.append(diff_sig, 0)

        # Step 3: Square
        squared = diff_sig ** 2

        # Step 4: Moving window integration (150 ms)
        win_len = max(int(0.150 * fs), 3)
        integrator = np.ones(win_len) / float(win_len)
        integrated = np.convolve(squared, integrator, mode="same")

        # Step 5: Adaptive threshold and peak detection
        threshold_I = 0.50 * np.max(integrated)
        min_distance = max(int(0.40 * fs), 1)

        candidates, properties = signal.find_peaks(
            integrated,
            height=threshold_I * 0.3,
            distance=min_distance,
            prominence=threshold_I * 0.2,
        )

        if len(candidates) == 0:
            return _detect_r_peaks_fallback(ecg, fs)

        # Adaptive threshold refinement
        peak_heights = integrated[candidates]
        signal_peaks = []
        spki = 0.0
        npki = 0.0

        for i, cand in enumerate(candidates):
            h = peak_heights[i]
            if i == 0:
                spki = h
                signal_peaks.append(cand)
                continue
            threshold = npki + 0.25 * (spki - npki)
            if h > threshold:
                signal_peaks.append(cand)
                spki = 0.125 * h + 0.875 * spki
            else:
                npki = 0.125 * h + 0.875 * npki

        if len(signal_peaks) == 0:
            return _detect_r_peaks_fallback(ecg, fs)

        # Step 6: Refractory period (200 ms)
        refractory = max(int(0.200 * fs), 1)
        filtered_peaks = [signal_peaks[0]]
        for pk in signal_peaks[1:]:
            if pk - filtered_peaks[-1] >= refractory:
                filtered_peaks.append(pk)

        # Step 7: Refine to true peak in original signal (+/-50 ms)
        search_half = max(int(0.050 * fs), 1)
        abs_ecg = np.abs(ecg - np.median(ecg))

        refined_peaks = []
        for pk in filtered_peaks:
            lo = max(0, pk - search_half)
            hi = min(len(ecg), pk + search_half + 1)
            local_peak = lo + int(np.argmax(abs_ecg[lo:hi]))
            refined_peaks.append(local_peak)

        refined_peaks = np.array(refined_peaks, dtype=int)

        # Step 8: Validate RR intervals (reject physiologically impossible)
        if len(refined_peaks) >= 2:
            valid = [refined_peaks[0]]
            for i in range(1, len(refined_peaks)):
                rr_sec = (refined_peaks[i] - valid[-1]) / fs
                if 0.300 <= rr_sec <= 2.000:
                    valid.append(refined_peaks[i])
            refined_peaks = np.array(valid, dtype=int)

        return refined_peaks

    except Exception as exc:
        log.debug(f"Pan-Tompkins failed: {exc}")
        return _detect_r_peaks_fallback(ecg, fs)


def compute_heart_rate(ecg: np.ndarray,
                       fs:  float) -> Tuple[float, float, float]:
    """
    Return (mean_hr, min_hr, max_hr) in BPM from R-peak intervals.
    Returns (0, 0, 0) when fewer than 2 peaks are found.
    """
    peaks = detect_r_peaks(ecg, fs)
    if len(peaks) < 2:
        return 0.0, 0.0, 0.0

    rr  = np.diff(peaks) / fs
    hr  = 60.0 / rr
    hr  = hr[(hr >= 30) & (hr <= 200)]

    if len(hr) == 0:
        return 0.0, 0.0, 0.0
    return round(float(np.mean(hr)), 1), \
           round(float(np.min(hr)),  1), \
           round(float(np.max(hr)),  1)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI (standalone testing)
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    import pandas as pd

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    if len(sys.argv) < 2:
        print("Usage: python signal_processor.py digitized_ecg_data.csv")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    df = pd.read_csv(csv_path)
    leads_raw = df.groupby("lead")["mv"].apply(np.array).to_dict()
    fs = float(df["time_s"].diff().median() ** -1) if "time_s" in df.columns else FS_DEFAULT

    cleaned = process_leads(leads_raw, fs=fs)

    print(f"\n{'Lead':<12} {'Mean HR':>8} {'Min HR':>8} {'Max HR':>8}")
    print("-" * 40)
    for name, arr in cleaned.items():
        if name == "__fs__":
            continue
        mhr, lo, hi = compute_heart_rate(arr, fs)
        print(f"{name:<12} {mhr:>8.1f} {lo:>8.1f} {hi:>8.1f}")
