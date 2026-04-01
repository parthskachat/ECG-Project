"""
app.py — CardioScan Pro  |  ECG Analytics Dashboard
Clean ECG display. No calibration pulse. No square waves.
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="CardioScan Pro",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR.parent / "data"
CSV_DIR    = DATA_DIR / "csvs"
META_FILE  = DATA_DIR / "patients.json"

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');
  :root {
    --bg-primary:   #070B12;
    --bg-card:      #0D1320;
    --bg-card2:     #111826;
    --accent:       #E63946;
    --accent2:      #00D4C8;
    --accent3:      #F4A261;
    --text-primary: #EDF2F7;
    --text-muted:   #718096;
    --border:       #1E2D42;
  }
  .stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    font-family: 'Sora', sans-serif !important;
    color: var(--text-primary) !important;
  }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090E18 0%, #0A1220 100%) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] * { color: var(--text-primary) !important; }
  [data-testid="stSidebarNav"] { display: none; }
  .block-container { padding-top: 1rem !important; max-width: 100% !important; }
  .metric-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px 24px; position: relative; overflow: hidden;
  }
  .metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
  .metric-card.red::before   { background: var(--accent); }
  .metric-card.teal::before  { background: var(--accent2); }
  .metric-card.amber::before { background: var(--accent3); }
  .metric-card.blue::before  { background: #4A90D9; }
  .metric-label { font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: 2px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px; }
  .metric-value { font-family: 'Space Mono', monospace; font-size: 36px; font-weight: 700; line-height: 1; margin-bottom: 4px; }
  .metric-unit { font-size: 13px; color: var(--text-muted); font-weight: 300; }
  .metric-badge { display: inline-block; font-size: 10px; letter-spacing: 1px; padding: 2px 8px; border-radius: 20px; margin-top: 6px; font-family: 'Space Mono', monospace; }
  .badge-normal  { background: rgba(0,212,200,0.12); color: var(--accent2); border: 1px solid rgba(0,212,200,0.25); }
  .badge-warning { background: rgba(244,162,97,0.12); color: var(--accent3); border: 1px solid rgba(244,162,97,0.25); }
  .badge-alert   { background: rgba(230,57,70,0.12);  color: var(--accent);  border: 1px solid rgba(230,57,70,0.25); }
  .section-header { font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 3px; color: var(--text-muted); text-transform: uppercase; border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-bottom: 16px; margin-top: 8px; }
  .patient-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
  .patient-name { font-size: 20px; font-weight: 700; color: var(--text-primary); margin-bottom: 4px; }
  .patient-meta { font-size: 12px; color: var(--text-muted); font-family: 'Space Mono', monospace; }
  .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite; }
  .status-ready   { background: var(--accent2); box-shadow: 0 0 8px var(--accent2); }
  .status-process { background: var(--accent3); box-shadow: 0 0 8px var(--accent3); }
  .status-error   { background: var(--accent);  box-shadow: 0 0 8px var(--accent); }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  .interp-box { background: linear-gradient(135deg, #0D1320, #111826); border: 1px solid var(--border); border-left: 3px solid var(--accent2); border-radius: 8px; padding: 20px 24px; font-size: 14px; line-height: 1.8; color: #CBD5E0; }
  .interp-box .header { font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: 2px; color: var(--accent2); text-transform: uppercase; margin-bottom: 12px; }
  .chart-wrap { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 4px; margin-bottom: 8px; }
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: var(--bg-primary); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .top-header { display: flex; align-items: center; gap: 16px; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }
  .logo-mark { font-family: 'Space Mono', monospace; font-size: 22px; font-weight: 700; color: var(--accent); letter-spacing: -1px; }
  .logo-sub { font-size: 10px; letter-spacing: 4px; color: var(--text-muted); text-transform: uppercase; }
  .live-badge { margin-left: auto; font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: 2px; color: var(--accent2); border: 1px solid rgba(0,212,200,0.3); padding: 4px 12px; border-radius: 20px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=5)
def load_registry() -> list:
    if META_FILE.exists():
        return json.loads(META_FILE.read_text())
    return []


def load_csv(record_id: str) -> pd.DataFrame | None:
    csv_path = CSV_DIR / f"{record_id}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def hr_badge(hr) -> str:
    if hr is None or hr == 0:
        return '<span class="metric-badge badge-warning">N/A</span>'
    hr = float(hr)
    if 60 <= hr <= 100:
        return '<span class="metric-badge badge-normal">NORMAL</span>'
    elif hr < 60:
        return '<span class="metric-badge badge-warning">BRADYCARDIA</span>'
    else:
        return '<span class="metric-badge badge-alert">TACHYCARDIA</span>'


def interval_badge(val, lo, hi) -> str:
    if val is None:
        return '<span class="metric-badge badge-warning">N/A</span>'
    if lo <= float(val) <= hi:
        return '<span class="metric-badge badge-normal">NORMAL</span>'
    return '<span class="metric-badge badge-alert">ABNORMAL</span>'


def status_html(status: str) -> str:
    cls = {"ready": "status-ready", "processing": "status-process", "error": "status-error"}.get(status, "status-process")
    return f'<span class="status-dot {cls}"></span>{status.upper()}'


LEAD_COLORS = {
    "I": "#E63946", "II": "#00D4C8", "III": "#F4A261",
    "aVR": "#A78BFA", "aVL": "#34D399", "aVF": "#FCD34D",
    "V1": "#F87171", "V2": "#60A5FA", "V3": "#A3E635",
    "V4": "#FB923C", "V5": "#E879F9", "V6": "#38BDF8",
    "II_rhythm": "#00D4C8",
}


def hex_to_rgba(hex_color: str, opacity: float = 0.10) -> str:
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(ch * 2 for ch in hex_color)
    if len(hex_color) != 6:
        return f"rgba(255, 255, 255, {opacity:.2f})"
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {opacity:.2f})"
    except ValueError:
        return f"rgba(255, 255, 255, {opacity:.2f})"


def _build_ecg_grid_shapes(t_max: float, y_min: float, y_max: float,
                           lead_offsets: list = None) -> list:
    """Build ECG paper grid. Major every 0.2s/0.5mV, minor every 0.04s/0.1mV."""
    shapes = []

    # Vertical lines
    t_val = 0.0
    while t_val <= t_max + 0.01:
        is_major = abs(round(t_val, 3) % 0.2) < 0.005
        shapes.append(dict(
            type="line", x0=t_val, x1=t_val, y0=y_min, y1=y_max,
            xref="x", yref="y",
            line=dict(
                color="rgba(30, 45, 66, 0.85)" if is_major else "rgba(30, 45, 66, 0.30)",
                width=0.9 if is_major else 0.4,
            ), layer="below",
        ))
        t_val = round(t_val + 0.04, 4)

    # Horizontal lines
    if lead_offsets is not None:
        for offset in lead_offsets:
            v_lo = offset - 2.5
            v_hi = offset + 2.5
            v_val = round(np.floor(v_lo * 10) / 10.0, 1)
            while v_val <= v_hi:
                dist = abs(v_val - offset)
                is_major = abs(round(dist, 3) % 0.5) < 0.005
                shapes.append(dict(
                    type="line", x0=0, x1=t_max, y0=v_val, y1=v_val,
                    xref="x", yref="y",
                    line=dict(
                        color="rgba(30, 45, 66, 0.85)" if is_major else "rgba(30, 45, 66, 0.25)",
                        width=0.7 if is_major else 0.3,
                    ), layer="below",
                ))
                v_val = round(v_val + 0.1, 1)
    else:
        v_val = round(np.floor(y_min * 10) / 10.0, 1)
        while v_val <= y_max:
            is_major = abs(round(v_val, 3) % 0.5) < 0.005
            shapes.append(dict(
                type="line", x0=0, x1=t_max, y0=v_val, y1=v_val,
                xref="x", yref="y",
                line=dict(
                    color="rgba(30, 45, 66, 0.85)" if is_major else "rgba(30, 45, 66, 0.25)",
                    width=0.7 if is_major else 0.3,
                ), layer="below",
            ))
            v_val = round(v_val + 0.1, 1)

    if len(shapes) > 4000:
        shapes = [s for s in shapes if s["line"]["width"] >= 0.7]

    return shapes


def make_ecg_chart(signals: dict, fs: float = 299.0, title: str = "") -> go.Figure:
    """Stacked multi-lead ECG chart. No calibration pulse."""
    leads = list(signals.keys())
    n = len(leads)

    global_ptp = max((np.ptp(sig) for sig in signals.values()), default=3.0)
    offset_step = max(3.0, (global_ptp + 1.0) * 1.3)

    fig = go.Figure()
    max_duration = 0.0
    lead_offsets = []

    for i, lead_name in enumerate(leads):
        sig = signals[lead_name]
        offset = (n - 1 - i) * offset_step
        lead_offsets.append(offset)
        color = LEAD_COLORS.get(lead_name, "#FFFFFF")
        t = np.arange(len(sig)) / fs

        if t[-1] > max_duration:
            max_duration = t[-1]

        fig.add_trace(go.Scattergl(
            x=t, y=sig + offset, mode="lines", name=lead_name,
            line=dict(color=color, width=1.0),
            hovertemplate=f"<b>{lead_name}</b><br>Time: %{{x:.3f}}s<br>mV: %{{customdata:.3f}}<extra></extra>",
            customdata=sig,
        ))

        fig.add_annotation(
            x=0, y=offset, xref="paper", yref="y",
            text=f"<b>{lead_name}</b>", showarrow=False, xanchor="right",
            font=dict(size=10, color=color, family="Space Mono"),
            bgcolor="rgba(7,11,18,0.7)", borderpad=2,
        )

    total_duration = max_duration + 0.1
    y_min = -offset_step * 0.5
    y_max = (n - 1) * offset_step + offset_step * 0.5 + 1.5

    shapes = _build_ecg_grid_shapes(total_duration, y_min, y_max, lead_offsets=lead_offsets)

    fig.update_layout(
        height=max(350, n * 80),
        margin=dict(l=60, r=20, t=30, b=40),
        paper_bgcolor="#0D1320", plot_bgcolor="#0A0F1A",
        font=dict(family="Sora", color="#718096"),
        title=dict(text=title, font=dict(size=12, color="#718096"), x=0.02),
        showlegend=True,
        legend=dict(orientation="h", x=0, y=-0.08, font=dict(size=9, family="Space Mono"), bgcolor="rgba(0,0,0,0)"),
        shapes=shapes,
        xaxis=dict(showgrid=False, zeroline=False, title=dict(text="Time (s)", font=dict(size=10)), tickfont=dict(size=9), dtick=0.2, range=[0, total_duration]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[y_min, y_max]),
        hovermode="x unified", dragmode="pan",
        modebar=dict(bgcolor="rgba(0,0,0,0)", color="#718096"),
    )
    return fig


def make_single_lead_chart(name: str, sig: np.ndarray, fs: float = 299.0) -> go.Figure:
    """Single-lead ECG chart. No calibration pulse."""
    color = LEAD_COLORS.get(name, "#E63946")
    t = np.arange(len(sig)) / fs
    total_duration = t[-1] + 0.05

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=t, y=sig, mode="lines", name=name,
        line=dict(color=color, width=1.2),
        fill="tozeroy", fillcolor=hex_to_rgba(color, opacity=0.08),
    ))

    sig_max = float(np.max(np.abs(sig))) + 0.5
    y_lo = -sig_max
    y_hi = sig_max + 0.5

    shapes = _build_ecg_grid_shapes(total_duration, y_lo, y_hi)

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=24, b=30),
        paper_bgcolor="#0D1320", plot_bgcolor="#0A0F1A",
        font=dict(family="Sora", color="#718096"),
        title=dict(text=f"<b>{name}</b>", font=dict(size=11, color=color), x=0.01),
        showlegend=False, shapes=shapes,
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=8), title=dict(text="Time (s)", font=dict(size=9)), dtick=0.2, range=[0, total_duration]),
        yaxis=dict(showgrid=False, zeroline=True, zerolinecolor="#1E2D42", zerolinewidth=1, tickfont=dict(size=8), title=dict(text="mV", font=dict(size=9)), range=[y_lo, y_hi]),
        dragmode="pan",
    )
    return fig


def generate_demo_ecg(duration: float = 5.0, fs: float = 299.0) -> dict:
    t = np.arange(0, duration, 1 / fs)
    leads = {}
    def ecg_wave(t, offset=0):
        hr = 72
        period = 60.0 / hr
        sig = np.zeros_like(t)
        for beat_start in np.arange(offset, duration, period):
            for tt in np.linspace(0, period, int(period * fs)):
                idx = int((beat_start + tt) * fs)
                if idx >= len(sig): break
                if 0.05 < tt < 0.15:
                    sig[idx] += 0.15 * np.sin(np.pi * (tt - 0.05) / 0.10)
                elif 0.18 < tt < 0.28:
                    if 0.18 < tt < 0.21:   sig[idx] -= 0.15
                    elif 0.21 < tt < 0.24: sig[idx] += 1.2 * np.sin(np.pi*(tt-0.21)/0.03)
                    else:                  sig[idx] -= 0.25
                elif 0.32 < tt < 0.48:
                    sig[idx] += 0.35 * np.sin(np.pi * (tt - 0.32) / 0.16)
        return sig + np.random.normal(0, 0.02, len(t))

    for i, name in enumerate(["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]):
        leads[name] = ecg_wave(t, offset=i * 0.01)
    return leads


DEMO_RECORD = {
    "id": "DEMO01", "patient_id": "DEMO", "patient_name": "Demo Patient",
    "age": 45, "gender": "M",
    "test_date": "2025-06-10T09:30:00Z", "received_at": "2025-06-10T09:31:22Z",
    "heart_rate": 72, "pr_interval": 158, "qt_interval": 394, "sample_rate_hz": 299,
    "interpretation": "Normal sinus rhythm. Heart rate 72 bpm. PR interval 158ms -- within normal limits. QRS duration 88ms. QTc 394ms -- normal. No ST segment elevation or depression. Impression: Normal 12-lead ECG.",
    "status": "ready",
}


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 24px">
      <div style="font-family:'Space Mono',monospace;font-size:18px;font-weight:700;color:#E63946;letter-spacing:-0.5px;">CardioScan Pro</div>
      <div style="font-size:10px;letter-spacing:3px;color:#718096;text-transform:uppercase;margin-top:4px;">ECG Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Patient Records</div>', unsafe_allow_html=True)

    registry = load_registry()
    if not registry:
        registry = [DEMO_RECORD]
        use_demo = True
    else:
        use_demo = False

    def _record_label(r: dict) -> str:
        name = r.get("patient_name", "Unknown Patient")
        date = (r.get("test_date") or r.get("timestamp") or "")[:10] or "No Date"
        rec_id = r.get("id", r.get("patient_id", "?"))[:6]
        return f"{name} -- {date} [{rec_id}]"

    record_options = {_record_label(r): r for r in registry}
    selected_label = st.selectbox("Select Patient", options=list(record_options.keys()), label_visibility="collapsed")
    record = record_options[selected_label]

    st.markdown('<div class="section-header" style="margin-top:24px">Display Options</div>', unsafe_allow_html=True)

    all_leads = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6","II_rhythm"]
    selected_leads = st.multiselect("Leads to display", options=all_leads, default=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"])
    view_mode = st.radio("Chart layout", ["Stacked (all leads)", "Grid (2 columns)", "Individual"], index=0)

    fs_default = int(record.get("sample_rate_hz", 299))
    fs_input = st.number_input("Sample Rate (Hz)", min_value=100, max_value=2000, value=fs_default, step=1)

    if use_demo:
        st.info("No webhook data yet. Showing demo patient.", icon="ℹ️")

    st.markdown("---")
    st.markdown(f'<div style="font-size:10px;color:#4A5568;font-family:Space Mono,monospace;">Last refresh: {datetime.now().strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Main content ───────────────────────────────────────────────────────────

col_logo, col_live = st.columns([3, 1])
with col_logo:
    st.markdown("""
    <div class="top-header">
      <div><div class="logo-mark">CardioScan Pro</div><div class="logo-sub">12-Lead ECG Diagnostic Suite</div></div>
      <div class="live-badge">LIVE MONITORING</div>
    </div>
    """, unsafe_allow_html=True)

col_pat, col_hr, col_pr, col_qt, col_status = st.columns([2.2, 1, 1, 1, 1])

with col_pat:
    gender_icon = "M" if record.get("gender","") in ("M","male","Male") else "F"
    _pat_name = record.get("patient_name", "Unknown Patient")
    _pat_id = record.get("patient_id", record.get("id", "--"))
    _date_str = (record.get("test_date") or record.get("timestamp") or "")[:10] or "--"
    _recv_str = (record.get("received_at") or "")[:19].replace("T", " ") or "--"
    st.markdown(f"""
    <div class="patient-card">
      <div class="patient-name">{_pat_name}</div>
      <div class="patient-meta">{gender_icon} {record.get('age','--')} yrs | ID: {_pat_id} | {status_html(record.get('status','processing'))}</div>
      <div class="patient-meta" style="margin-top:8px;">Date: {_date_str} Received: {_recv_str}</div>
    </div>
    """, unsafe_allow_html=True)

with col_hr:
    hr = record.get("heart_rate", 0)
    st.markdown(f'<div class="metric-card red"><div class="metric-label">Heart Rate</div><div class="metric-value" style="color:#E63946">{hr or "--"}</div><div class="metric-unit">BPM</div>{hr_badge(hr)}</div>', unsafe_allow_html=True)

with col_pr:
    pr = record.get("pr_interval")
    st.markdown(f'<div class="metric-card teal"><div class="metric-label">PR Interval</div><div class="metric-value" style="color:#00D4C8">{pr or "--"}</div><div class="metric-unit">ms</div>{interval_badge(pr, 120, 200)}</div>', unsafe_allow_html=True)

with col_qt:
    qt = record.get("qt_interval")
    st.markdown(f'<div class="metric-card amber"><div class="metric-label">QT Interval</div><div class="metric-value" style="color:#F4A261">{qt or "--"}</div><div class="metric-unit">ms</div>{interval_badge(qt, 350, 450)}</div>', unsafe_allow_html=True)

with col_status:
    n_records = len(registry)
    ready = sum(1 for r in registry if r.get("status") == "ready")
    st.markdown(f'<div class="metric-card blue"><div class="metric-label">Records</div><div class="metric-value" style="color:#4A90D9">{n_records}</div><div class="metric-unit">Total / {ready} ready</div><span class="metric-badge badge-normal">SYSTEM OK</span></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Load ECG data ──
if use_demo:
    all_signals = generate_demo_ecg(fs=float(fs_input))
elif record.get("status") == "ready":
    record_id = record.get("id", "")
    df = load_csv(record_id) if record_id else None
    if df is not None:
        all_signals = {lead: grp["mv"].values for lead, grp in df.groupby("lead")}
    else:
        st.warning("CSV not found -- showing synthetic preview.", icon="⚠️")
        all_signals = generate_demo_ecg(fs=float(fs_input))
elif record.get("status") == "processing":
    st.info("Record still processing -- showing synthetic preview.", icon="ℹ️")
    all_signals = generate_demo_ecg(fs=float(fs_input))
else:
    all_signals = None

# ── ECG Leads Section ──
st.markdown('<div class="section-header">ECG WAVEFORMS</div>', unsafe_allow_html=True)

if all_signals:
    signals_to_show = {k: v for k, v in all_signals.items() if k in selected_leads}

    if not signals_to_show:
        st.info("Select at least one lead from the sidebar.")
    else:
        if view_mode == "Stacked (all leads)":
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            fig = make_ecg_chart(signals_to_show, fs=float(fs_input), title="12-Lead ECG -- Interactive (drag to pan, scroll to zoom)")
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True, "modeBarButtonsToRemove": ["select2d", "lasso2d"], "toImageButtonOptions": {"format": "png", "scale": 2}})
            st.markdown('</div>', unsafe_allow_html=True)

        elif view_mode == "Grid (2 columns)":
            leads_list = list(signals_to_show.keys())
            for i in range(0, len(leads_list), 2):
                cols = st.columns(2)
                for j, lead in enumerate(leads_list[i:i+2]):
                    with cols[j]:
                        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                        fig = make_single_lead_chart(lead, signals_to_show[lead], float(fs_input))
                        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": False})
                        st.markdown('</div>', unsafe_allow_html=True)

        else:
            for lead, sig in signals_to_show.items():
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                fig = make_single_lead_chart(lead, sig, float(fs_input))
                st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

    if "II_rhythm" in all_signals and "II_rhythm" in selected_leads:
        st.markdown('<div class="section-header" style="margin-top:8px">RHYTHM STRIP -- LEAD II</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        fig_rhythm = make_single_lead_chart("II_rhythm", all_signals["II_rhythm"], float(fs_input))
        fig_rhythm.update_layout(height=140)
        st.plotly_chart(fig_rhythm, use_container_width=True, config={"scrollZoom": True, "displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error(f"ECG data unavailable (status: {record.get('status', 'unknown')}).")

# ── Interpretation ──
st.markdown('<div class="section-header" style="margin-top:8px">PHYSICIAN INTERPRETATION</div>', unsafe_allow_html=True)

interpretation = record.get("interpretation", "")
if interpretation:
    st.markdown(f'<div class="interp-box"><div class="header">Final Report -- Verified by Tricog Cardiologist</div>{interpretation}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="interp-box"><div class="header">Pending</div>Interpretation not yet available.</div>', unsafe_allow_html=True)

# ── Raw data ──
if all_signals:
    with st.expander("Raw Signal Data (CSV Preview)", expanded=False):
        rows = []
        for lead, sig in list(all_signals.items())[:3]:
            for i, v in enumerate(sig[:500]):
                rows.append({"lead": lead, "sample": i, "time_s": round(i / float(fs_input), 4), "mv": round(v, 5)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=200)

st.markdown('<div style="margin-top:48px;padding-top:16px;border-top:1px solid #1E2D42;text-align:center;font-size:10px;color:#4A5568;font-family:Space Mono,monospace;letter-spacing:1px;">CARDIOSCAN PRO -- FOR CLINICAL USE ONLY -- ALWAYS VERIFY WITH TREATING PHYSICIAN</div>', unsafe_allow_html=True)
