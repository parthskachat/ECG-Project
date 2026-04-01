"""
digitizer.py — Clinical-Grade ECG Digitizer
============================================
Converts Tricog PDF ECG reports into per-lead mV signal arrays.

Architecture
------------
1. PDF → BGR image  (300 DPI via pdf2image / poppler)
2. Per-lead crop    (empirically-calibrated fractional coordinates)
3. Grid removal     (HSV masking + trace-protection inpaint)
4. Skeletonisation  (Zhang-Suen 1-px medial axis, preserves R-peak amplitude)
5. Pixel → mV       (FFT grid calibration → mm/px → mV/mm = 10 mm/mV standard)

Usage
-----
    from digitizer import digitize_pdf
    data = digitize_pdf(Path('r.pdf'))
"""

import logging
import warnings
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from pdf2image import convert_from_path
from scipy.signal import find_peaks, savgol_filter
from skimage.morphology import skeletonize as _sk_skeletonize

warnings.filterwarnings("ignore")
log = logging.getLogger("ecg.digitizer")

# ── Physical constants ────────────────────────────────────────────────────────
DPI              = 300
PAPER_SPEED      = 25.0
MM_PER_MV        = 10.0
PX_PER_MM_THEORY = DPI / 25.4

# ── Lead layout ───────────────────────────────────────────────────────────────
LEAD_LAYOUT: Dict[str, Tuple[float, float, float, float]] = {
    "I":         (0.1959, 0.3329, 0.0869, 0.2916),
    "aVR":       (0.1959, 0.3329, 0.2922, 0.4971),
    "V1":        (0.1959, 0.3329, 0.4977, 0.7024),
    "V4":        (0.1959, 0.3329, 0.7030, 0.9079),
    "II":        (0.3414, 0.4784, 0.0869, 0.2916),
    "aVL":       (0.3414, 0.4784, 0.2922, 0.4971),
    "V2":        (0.3414, 0.4784, 0.4977, 0.7024),
    "V5":        (0.3414, 0.4784, 0.7030, 0.9079),
    "III":       (0.4853, 0.6223, 0.0869, 0.2916),
    "aVF":       (0.4853, 0.6223, 0.2922, 0.4971),
    "V3":        (0.4853, 0.6223, 0.4977, 0.7024),
    "V6":        (0.4853, 0.6223, 0.7030, 0.9079),
    "II_rhythm": (0.6312, 0.7682, 0.0869, 0.9079),
}

LEAD_ORDER = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3",
    "V4", "V5", "V6",
    "II_rhythm",
]

# ── Grid-removal thresholds ───────────────────────────────────────────────────
_TRACE_DARK_THRESH = 110
_PROTECT_KERNEL    = 17
_INPAINT_RADIUS    = 3


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PDF to image
# ═══════════════════════════════════════════════════════════════════════════════

def pdf_to_image(pdf_path: Path) -> np.ndarray:
    """Render first page of PDF at DPI, return BGR uint8 array."""
    pages = convert_from_path(
        str(pdf_path), dpi=DPI, first_page=1, last_page=1, fmt="jpeg"
    )
    return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Grid removal
# ═══════════════════════════════════════════════════════════════════════════════

def _build_grid_mask(bgr: np.ndarray) -> np.ndarray:
    """Create binary mask of red/pink ECG grid pixels using HSV ranges."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m_red_lo = cv2.inRange(hsv,
                           np.array([0,   40, 160], np.uint8),
                           np.array([15, 255, 255], np.uint8))
    m_red_hi = cv2.inRange(hsv,
                           np.array([165,  40, 160], np.uint8),
                           np.array([179, 255, 255], np.uint8))
    m_pink   = cv2.inRange(hsv,
                           np.array([0,   8, 210], np.uint8),
                           np.array([20, 70, 255], np.uint8))
    grid_mask = m_red_lo | m_red_hi | m_pink
    return cv2.dilate(grid_mask, np.ones((3, 3), np.uint8))


def _build_trace_protection(gray: np.ndarray) -> np.ndarray:
    """Dilate dark (trace) pixels into a protection zone."""
    _, dark = cv2.threshold(gray, _TRACE_DARK_THRESH, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_PROTECT_KERNEL, _PROTECT_KERNEL)
    )
    return cv2.dilate(dark, kernel)


def remove_grid(bgr_crop: np.ndarray) -> np.ndarray:
    """Remove ECG grid from a single-lead BGR crop. Return grayscale."""
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    grid_mask   = _build_grid_mask(bgr_crop)
    protect     = _build_trace_protection(gray)
    inpaint_mask = cv2.bitwise_and(grid_mask, cv2.bitwise_not(protect))
    cleaned_bgr = cv2.inpaint(bgr_crop, inpaint_mask,
                              _INPAINT_RADIUS, cv2.INPAINT_NS)
    return cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Grid calibration (FFT-based px/mm estimation)
# ═══════════════════════════════════════════════════════════════════════════════

def _fft_pitch(profile_1d: np.ndarray) -> float:
    """Estimate dominant spatial period (pixels) from a 1-D projection."""
    n = len(profile_1d)
    if n < 32:
        return PX_PER_MM_THEORY
    arr = (profile_1d - profile_1d.mean()) * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(arr))
    spectrum[:3] = 0
    freqs = np.fft.rfftfreq(n)
    peak = np.argmax(spectrum)
    if peak == 0 or freqs[peak] < 1e-9:
        return PX_PER_MM_THEORY
    return float(1.0 / freqs[peak])


def calibrate_grid(gray_crop: np.ndarray) -> Tuple[float, float]:
    """Estimate px/mm in X (time) and Y (amplitude) from the ECG grid."""
    def pitch_to_ppmm(pitch: float) -> float:
        if   8 < pitch < 20:  return pitch
        elif 45 < pitch < 75: return pitch / 5.0
        else:                  return PX_PER_MM_THEORY

    px_x = pitch_to_ppmm(_fft_pitch(gray_crop.mean(axis=0)))
    px_y = pitch_to_ppmm(_fft_pitch(gray_crop.mean(axis=1)))
    px_x = float(np.clip(px_x, 9.0, 15.0))
    px_y = float(np.clip(px_y, 9.0, 15.0))
    return px_x, px_y


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Signal extraction (skeletonisation + continuity-aware tracker)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_signal_px(gray_crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract the 1-D Y-position signal (pixel rows) from a clean gray crop.

    Uses sub-pixel weighted centroid, continuity-aware gap filling,
    and selective Savitzky-Golay smoothing on gap-filled segments only.
    """
    ch, cw = gray_crop.shape

    # Binary threshold
    _, binary = cv2.threshold(
        gray_crop, _TRACE_DARK_THRESH, 255, cv2.THRESH_BINARY_INV
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if binary.sum() < cw * 4:
        return None

    # Skeletonise to 1-px medial axis
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skel = _sk_skeletonize(binary > 0).astype(np.uint8) * 255

    # Per-column extraction with sub-pixel centroid
    sig_px = np.full(cw, np.nan)
    source = np.zeros(cw, dtype=np.int8)  # 0=missing, 1=skeleton, 2=binary, 3=gap-filled
    half = ch // 2

    for x in range(cw):
        skel_rows_center = np.where(skel[:, x] > 0)[0]

        if len(skel_rows_center) > 0:
            # Sub-pixel centroid from skeleton pixels in column x and neighbors
            all_rows = []
            all_weights = []
            for dx in [-1, 0, 1]:
                nx = x + dx
                if 0 <= nx < cw:
                    rows_dx = np.where(skel[:, nx] > 0)[0]
                    if len(rows_dx) > 0:
                        w = 2.0 if dx == 0 else 1.0
                        all_rows.extend(rows_dx.tolist())
                        all_weights.extend([w] * len(rows_dx))

            all_rows = np.array(all_rows, dtype=float)
            all_weights = np.array(all_weights, dtype=float)

            # Filter to rows near the anchor (avoid skeleton branches)
            anchor = skel_rows_center[np.argmin(np.abs(skel_rows_center - half))]
            nearby_mask = np.abs(all_rows - anchor) < 10
            if nearby_mask.sum() > 0:
                all_rows = all_rows[nearby_mask]
                all_weights = all_weights[nearby_mask]

            sig_px[x] = float(np.average(all_rows, weights=all_weights))
            source[x] = 1
        else:
            # Fallback: weighted centroid of binary trace pixels
            bin_rows = np.where(binary[:, x] > 0)[0]
            if len(bin_rows) > 0:
                intensities = gray_crop[bin_rows, x].astype(float)
                weights = ((_TRACE_DARK_THRESH + 1.0) - intensities)
                weights = np.maximum(weights, 1.0)
                closest_to_half = bin_rows[np.argmin(np.abs(bin_rows - half))]
                nearby = np.abs(bin_rows - closest_to_half) < 15
                if nearby.sum() > 0:
                    sig_px[x] = float(np.average(bin_rows[nearby], weights=weights[nearby]))
                else:
                    sig_px[x] = float(np.average(bin_rows, weights=weights))
                source[x] = 2

    valid_mask = ~np.isnan(sig_px)
    valid_frac = valid_mask.sum() / cw
    if valid_frac < 0.05:
        return None

    # ── Continuity-aware gap filling ──────────────────────────────────────
    HISTORY = 5

    gap_starts = []
    gap_ends = []
    in_gap = False
    for x in range(cw):
        if np.isnan(sig_px[x]) and not in_gap:
            in_gap = True
            gap_starts.append(x)
        elif not np.isnan(sig_px[x]) and in_gap:
            in_gap = False
            gap_ends.append(x)
    if in_gap:
        gap_ends.append(cw)

    for gs, ge in zip(gap_starts, gap_ends):
        gap_len = ge - gs

        # Collect history before gap
        before_x = []
        before_y = []
        bx = gs - 1
        while bx >= 0 and len(before_x) < HISTORY:
            if not np.isnan(sig_px[bx]):
                before_x.append(bx)
                before_y.append(sig_px[bx])
            bx -= 1
        before_x.reverse()
        before_y.reverse()

        # Collect history after gap
        after_x = []
        after_y = []
        ax = ge
        while ax < cw and len(after_x) < HISTORY:
            if not np.isnan(sig_px[ax]):
                after_x.append(ax)
                after_y.append(sig_px[ax])
            ax += 1

        if len(before_x) == 0 and len(after_x) == 0:
            sig_px[gs:ge] = float(half)
            source[gs:ge] = 3
            continue

        if len(before_x) == 0:
            sig_px[gs:ge] = after_y[0]
            source[gs:ge] = 3
            continue

        if len(after_x) == 0:
            sig_px[gs:ge] = before_y[-1]
            source[gs:ge] = 3
            continue

        # Short gaps: continuity-aware search
        if gap_len <= 5:
            for x in range(gs, ge):
                if len(before_x) >= 2:
                    dx_val = before_x[-1] - before_x[-2]
                    dy_val = before_y[-1] - before_y[-2]
                    slope = dy_val / dx_val if dx_val != 0 else 0.0
                    predicted = before_y[-1] + slope * (x - before_x[-1])
                else:
                    predicted = before_y[-1]

                predicted = np.clip(predicted, 0, ch - 1)

                search_radius = 5
                row_lo = max(0, int(predicted) - search_radius)
                row_hi = min(ch, int(predicted) + search_radius + 1)
                bin_rows = np.where(binary[row_lo:row_hi, x] > 0)[0] + row_lo

                if len(bin_rows) > 0:
                    intensities = gray_crop[bin_rows, x].astype(float)
                    weights = ((_TRACE_DARK_THRESH + 1.0) - intensities)
                    weights = np.maximum(weights, 1.0)
                    sig_px[x] = float(np.average(bin_rows, weights=weights))
                else:
                    sig_px[x] = predicted

                source[x] = 3
                before_x.append(x)
                before_y.append(sig_px[x])
                if len(before_x) > HISTORY:
                    before_x.pop(0)
                    before_y.pop(0)
        else:
            # Longer gaps: linear interpolation between boundary values
            all_anchor_x = before_x + after_x
            all_anchor_y = before_y + after_y
            for x in range(gs, ge):
                sig_px[x] = np.interp(x, all_anchor_x, all_anchor_y)
                source[x] = 3

    # Final gap fill for any remaining NaN
    still_nan = np.isnan(sig_px)
    if still_nan.any():
        valid_x = np.where(~still_nan)[0]
        if len(valid_x) >= 2:
            sig_px = np.interp(np.arange(cw), valid_x, sig_px[valid_x])
            source[still_nan] = 3
        elif len(valid_x) == 1:
            sig_px[:] = sig_px[valid_x[0]]
            source[still_nan] = 3
        else:
            return None

    # ── Selective Savitzky-Golay smoothing on gap-filled segments only ────
    gap_filled_mask = (source == 3)

    if gap_filled_mask.sum() > 0:
        sg_window = 11
        sg_poly = 3

        gap_runs = []
        in_run = False
        for x in range(cw):
            if gap_filled_mask[x] and not in_run:
                in_run = True
                run_start = x
            elif not gap_filled_mask[x] and in_run:
                in_run = False
                gap_runs.append((run_start, x))
        if in_run:
            gap_runs.append((run_start, cw))

        for rs, re in gap_runs:
            ext = sg_window // 2
            smooth_start = max(0, rs - ext)
            smooth_end = min(cw, re + ext)
            seg_len = smooth_end - smooth_start

            if seg_len >= sg_window:
                smoothed = savgol_filter(sig_px[smooth_start:smooth_end],
                                         sg_window, sg_poly)
                for x in range(rs, re):
                    sig_px[x] = smoothed[x - smooth_start]

    return sig_px


def px_to_mv(sig_px: np.ndarray, px_per_mm_y: float) -> np.ndarray:
    """Convert pixel-row signal to millivolts."""
    baseline = np.median(sig_px)
    delta_mm = (baseline - sig_px) / px_per_mm_y
    return delta_mm / MM_PER_MV


# ═══════════════════════════════════════════════════════════════════════════════
# 5. R-peak detector (verification / HR estimation)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_r_peaks_quick(sig_mv: np.ndarray, fs: float) -> np.ndarray:
    """Lightweight R-peak detector for signal quality checks."""
    if len(sig_mv) < int(0.5 * fs):
        return np.array([], dtype=int)
    try:
        abs_sig   = np.abs(sig_mv - sig_mv.mean())
        height_th = 0.30 * np.ptp(abs_sig)
        min_dist  = max(int(0.30 * fs), 1)
        peaks, _  = find_peaks(abs_sig, height=height_th, distance=min_dist)
        return peaks
    except Exception:
        return np.array([], dtype=int)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Vector PDF extraction (attempt before image processing)
# ═══════════════════════════════════════════════════════════════════════════════

def try_vector_extraction(pdf_path: Path) -> Optional[Dict[str, object]]:
    """
    Attempt to extract ECG signals directly from vector paths in the PDF.
    Returns dict of lead->mV arrays if successful, None if PDF is rasterized.
    """
    try:
        import pdfplumber
    except ImportError:
        log.warning("pdfplumber not installed -- skipping vector extraction check")
        return None

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if len(pdf.pages) == 0:
                log.debug("PDF has no pages")
                return None

            page = pdf.pages[0]
            page_width = float(page.width)
            page_height = float(page.height)

            curves = page.curves if hasattr(page, 'curves') and page.curves else []
            lines_list = page.lines if hasattr(page, 'lines') and page.lines else []

            total_vectors = len(curves) + len(lines_list)
            log.info(f"Vector check: {len(curves)} curves + {len(lines_list)} lines "
                     f"= {total_vectors} total vector objects")

            if total_vectors < 200:
                log.info("PDF appears to be rasterized (< 200 vector objects) -- "
                         "using image pipeline")
                return None

            log.info(f"Vector PDF detected ({total_vectors} objects) -- "
                     "attempting coordinate extraction")

            pt_to_px = DPI / 72.0
            all_points = []

            for curve in curves:
                pts = curve.get("pts", [])
                if len(pts) < 2:
                    continue
                path_points = []
                for pt in pts:
                    x_px = float(pt[0]) * pt_to_px
                    y_px = (page_height - float(pt[1])) * pt_to_px
                    path_points.append((x_px, y_px))
                if path_points:
                    all_points.append(path_points)

            for line_obj in lines_list:
                x0_l = float(line_obj.get("x0", 0)) * pt_to_px
                y0_l = (page_height - float(line_obj.get("y0", 0))) * pt_to_px
                x1_l = float(line_obj.get("x1", 0)) * pt_to_px
                y1_l = (page_height - float(line_obj.get("y1", 0))) * pt_to_px
                all_points.append([(x0_l, y0_l), (x1_l, y1_l)])

            if not all_points:
                log.warning("Vector objects found but no extractable points -- "
                            "falling back to image pipeline")
                return None

            img_h = page_height * pt_to_px
            img_w = page_width * pt_to_px

            results: Dict[str, object] = {}
            fs_estimates: List[float] = []

            for lead_name in LEAD_ORDER:
                if lead_name not in LEAD_LAYOUT:
                    continue

                r0, r1, c0, c1 = LEAD_LAYOUT[lead_name]
                y_top = r0 * img_h
                y_bot = r1 * img_h
                x_left = c0 * img_w
                x_right = c1 * img_w

                lead_points = []
                for path in all_points:
                    for (px_v, py_v) in path:
                        if x_left <= px_v <= x_right and y_top <= py_v <= y_bot:
                            lead_points.append((px_v, py_v))

                if len(lead_points) < 20:
                    continue

                lead_points.sort(key=lambda p_item: p_item[0])
                xs = np.array([p_item[0] for p_item in lead_points])
                ys = np.array([p_item[1] for p_item in lead_points])

                unique_xs, inv_idx = np.unique(xs, return_inverse=True)
                unique_ys = np.zeros_like(unique_xs)
                median_y_v = np.median(ys)
                for i in range(len(unique_xs)):
                    mask = inv_idx == i
                    candidates = ys[mask]
                    best = candidates[np.argmin(np.abs(candidates - median_y_v))]
                    unique_ys[i] = best

                xs = unique_xs
                ys = unique_ys

                if len(xs) < 10:
                    continue

                crop_width_px = x_right - x_left
                if lead_name == "II_rhythm":
                    expected_width_mm = 250.0
                else:
                    expected_width_mm = 62.5

                px_per_mm_x_v = crop_width_px / expected_width_mm
                px_per_mm_y_v = px_per_mm_x_v
                px_per_mm_x_v = float(np.clip(px_per_mm_x_v, 9.0, 15.0))
                px_per_mm_y_v = float(np.clip(px_per_mm_y_v, 9.0, 15.0))

                fs_lead = px_per_mm_x_v * PAPER_SPEED
                fs_estimates.append(fs_lead)

                baseline_y_val = np.median(ys)
                delta_mm = (baseline_y_val - ys) / px_per_mm_y_v
                sig_mv = delta_mm / MM_PER_MV

                n_samples = int(crop_width_px)
                x_uniform = np.linspace(xs[0], xs[-1], n_samples)
                sig_mv_uniform = np.interp(x_uniform, xs, sig_mv)

                results[lead_name] = sig_mv_uniform
                log.info(f"[{lead_name}] Vector extraction: {len(sig_mv_uniform)} samples  "
                         f"p-p={np.ptp(sig_mv_uniform):.3f} mV")

            if len(results) < 6:
                log.warning(f"Only {len(results)} leads extracted from vector data -- "
                            "insufficient, falling back to image pipeline")
                return None

            results["__fs__"] = float(np.median(fs_estimates)) if fs_estimates else 295.0
            log.info(f"Vector extraction complete: {len(results)-1}/13 leads  "
                     f"fs={results['__fs__']:.1f} Hz")
            return results

    except Exception as exc:
        log.warning(f"Vector extraction failed ({exc}) -- falling back to image pipeline")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def digitize_pdf(pdf_path: Path) -> Dict[str, object]:
    """
    End-to-end digitisation of a Tricog ECG PDF.

    Pipeline:
    1. Try vector extraction (pdfplumber) -- perfect accuracy if available
    2. Fall through to image pipeline if PDF is rasterized

    Returns
    -------
    dict
        Keys are lead names with np.ndarray values (mV), plus '__fs__' (Hz).
    """
    log.info(f"Digitising: {pdf_path.name}")

    # ── Step 0: Try vector extraction first ───────────────────────────────
    try:
        vector_result = try_vector_extraction(pdf_path)
        if vector_result is not None:
            log.info("Using vector-extracted data (perfect accuracy)")
            return vector_result
    except Exception as exc:
        log.warning(f"Vector extraction attempt failed ({exc})")

    log.info("PDF is rasterized -- using image processing pipeline")

    # ── Step 1: Render PDF to image ───────────────────────────────────────
    img_bgr = pdf_to_image(pdf_path)
    h_full, w_full = img_bgr.shape[:2]
    log.info(f"Rendered: {w_full}x{h_full} px at {DPI} DPI")

    results: Dict[str, object] = {}
    fs_estimates: List[float] = []

    # Pre-compute grayscale for adaptive crop scanning
    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Pre-compute baseline Y positions for all leads
    lead_baselines: Dict[str, int] = {}
    for lead_name in LEAD_ORDER:
        if lead_name not in LEAD_LAYOUT:
            continue
        r0, r1, c0, c1 = LEAD_LAYOUT[lead_name]
        y0_orig = int(r0 * h_full)
        y1_orig = int(r1 * h_full)
        baseline_y = (y0_orig + y1_orig) // 2
        lead_baselines[lead_name] = baseline_y

    for lead_name in LEAD_ORDER:
        if lead_name not in LEAD_LAYOUT:
            continue

        r0, r1, c0, c1 = LEAD_LAYOUT[lead_name]
        y0_orig = int(r0 * h_full)
        y1_orig = int(r1 * h_full)
        x0 = int(c0 * w_full)
        x1 = int(c1 * w_full)
        baseline_y = lead_baselines[lead_name]
        orig_half = (y1_orig - y0_orig) // 2

        # ── Adaptive crop height ─────────────────────────────────────────
        scan_half = 250
        scan_y0 = max(0, baseline_y - scan_half)
        scan_y1 = min(h_full, baseline_y + scan_half)
        scan_strip = gray_full[scan_y0:scan_y1, x0:x1]

        dark_mask_scan = scan_strip < _TRACE_DARK_THRESH
        dark_rows = np.where(dark_mask_scan.any(axis=1))[0]

        if len(dark_rows) > 10:
            abs_dark_rows = dark_rows + scan_y0
            distances = np.abs(abs_dark_rows - baseline_y)
            actual_extent = int(np.percentile(distances, 99))
            crop_half = max(orig_half, actual_extent + 25)
        else:
            crop_half = orig_half

        # Prevent bleeding into adjacent leads
        max_allowed_half = scan_half
        for other_name, other_baseline in lead_baselines.items():
            if other_name == lead_name:
                continue
            other_r0, other_r1, other_c0, other_c1 = LEAD_LAYOUT[other_name]
            other_x0 = int(other_c0 * w_full)
            other_x1 = int(other_c1 * w_full)
            if other_x1 > x0 and other_x0 < x1:
                separation = abs(other_baseline - baseline_y)
                if separation > 0:
                    allowed = (separation // 2) - 20
                    if allowed > 0:
                        max_allowed_half = min(max_allowed_half, allowed)

        crop_half = min(crop_half, max_allowed_half)
        crop_half = max(crop_half, 100)

        y0 = max(0, baseline_y - crop_half)
        y1 = min(h_full, baseline_y + crop_half)

        if crop_half != orig_half:
            log.debug(f"[{lead_name}] Adaptive crop: +-{crop_half}px "
                      f"(was +-{orig_half}px)")

        crop_bgr = img_bgr[y0:y1, x0:x1]
        ch, cw = crop_bgr.shape[:2]

        if ch < 20 or cw < 20:
            log.warning(f"[{lead_name}] Crop too small ({cw}x{ch}), skipping")
            continue

        # ── Grid removal ────────────────────────────────────────────────────
        try:
            gray_clean = remove_grid(crop_bgr)
        except Exception as exc:
            log.warning(f"[{lead_name}] Grid removal failed ({exc}), using raw gray")
            gray_clean = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # ── Grid calibration ────────────────────────────────────────────────
        px_per_mm_x, px_per_mm_y = calibrate_grid(
            cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        )
        fs_lead = px_per_mm_x * PAPER_SPEED
        fs_estimates.append(fs_lead)
        log.debug(f"[{lead_name}] px/mm x={px_per_mm_x:.3f} y={px_per_mm_y:.3f} "
                  f"fs={fs_lead:.1f} Hz")

        # ── Signal extraction ────────────────────────────────────────────────
        sig_px = extract_signal_px(gray_clean)
        if sig_px is None:
            log.warning(f"[{lead_name}] No trace found in crop")
            continue

        sig_mv = px_to_mv(sig_px, px_per_mm_y)

        # ── Quality report ───────────────────────────────────────────────────
        r_peaks = detect_r_peaks_quick(sig_mv, fs_lead)
        if len(r_peaks) >= 2:
            rr = np.diff(r_peaks) / fs_lead
            hr = round(60.0 / np.mean(rr), 1)
            log.info(f"[{lead_name}] {len(r_peaks)} peaks  HR~{hr} bpm  "
                     f"p-p={np.ptp(sig_mv):.3f} mV")
        elif len(r_peaks) == 1:
            log.info(f"[{lead_name}] 1 peak  p-p={np.ptp(sig_mv):.3f} mV")
        else:
            log.warning(f"[{lead_name}] No R-peaks detected  "
                        f"p-p={np.ptp(sig_mv):.3f} mV")

        results[lead_name] = sig_mv

    results["__fs__"] = float(np.median(fs_estimates)) if fs_estimates else 295.0
    log.info(f"Sample rate: {results['__fs__']:.1f} Hz  "
             f"Leads OK: {len(results)-1}/13")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Debug utility
# ═══════════════════════════════════════════════════════════════════════════════

def save_layout_debug(pdf_path: Path,
                      out_path: Path = Path("layout_debug.png")) -> None:
    """Save annotated image showing all crop rectangles."""
    img = pdf_to_image(pdf_path)
    h, w = img.shape[:2]
    vis  = img.copy()

    row_color = {
        "I": (0,255,0), "aVR": (0,255,0), "V1": (0,255,0), "V4": (0,255,0),
        "II": (0,200,255), "aVL": (0,200,255), "V2": (0,200,255), "V5": (0,200,255),
        "III": (0,100,255), "aVF": (0,100,255), "V3": (0,100,255), "V6": (0,100,255),
        "II_rhythm": (255, 0, 255),
    }

    for name, (r0, r1, c0, c1) in LEAD_LAYOUT.items():
        y0, y1 = int(r0*h), int(r1*h)
        x0, x1 = int(c0*w), int(c1*w)
        col = row_color.get(name, (255,255,255))
        cv2.rectangle(vis, (x0,y0), (x1,y1), col, 3)
        cv2.putText(vis, name, (x0+8, y0+35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)

    cv2.imwrite(str(out_path), vis)
    log.info(f"Layout debug saved -> {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python digitizer.py report.pdf")
        print("  python digitizer.py --debug report.pdf [layout.png]")
        sys.exit(1)

    if sys.argv[1] == "--debug":
        pdf = Path(sys.argv[2])
        out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("layout_debug.png")
        save_layout_debug(pdf, out)
        print(f"Saved -> {out}")
        sys.exit(0)

    pdf = Path(sys.argv[1])
    data = digitize_pdf(pdf)
    fs   = data.pop("__fs__")
    print(f"\n{'Lead':<12} {'Length':>8} {'Min mV':>8} {'Max mV':>8} {'p-p mV':>8}")
    print("-" * 48)
    for name, sig in data.items():
        print(f"{name:<12} {len(sig):>8} {sig.min():>8.3f} {sig.max():>8.3f} {np.ptp(sig):>8.3f}")
    print(f"\nSample rate: {fs:.1f} Hz")
