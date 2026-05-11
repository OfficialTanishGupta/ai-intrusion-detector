import streamlit as st
import pandas as pd
import random
import torch
import joblib
import time
import sys
import os

sys.path.append(os.path.abspath("../src"))
from preprocess import load_and_preprocess
from model_nn import Net


st.set_page_config(
    page_title="CyberShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
}

.stApp {
    background: #020b18;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,200,255,0.07) 0%, transparent 70%),
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(0,200,255,0.03) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(0,200,255,0.03) 40px);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050f1e 0%, #020b18 100%);
    border-right: 1px solid rgba(0,200,255,0.15);
}

[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] label {
    color: #00c8ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 0.05em;
}

/* ── Header / Title ── */
.cyber-header {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 24px 0 8px 0;
    border-bottom: 1px solid rgba(0,200,255,0.2);
    margin-bottom: 8px;
}
.cyber-header-icon {
    font-size: 3rem;
    filter: drop-shadow(0 0 12px #00c8ff);
}
.cyber-title {
    font-family: 'Exo 2', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    color: #ffffff;
    letter-spacing: 0.04em;
    line-height: 1.1;
    text-shadow: 0 0 24px rgba(0,200,255,0.5);
    margin: 0;
}
.cyber-subtitle {
    font-family: 'Share Tech Mono', monospace;
    color: #00c8ff;
    font-size: 0.78rem;
    letter-spacing: 0.15em;
    margin-top: 4px;
}
.cyber-badge {
    display: inline-block;
    background: rgba(0,200,255,0.08);
    border: 1px solid rgba(0,200,255,0.3);
    color: #00c8ff;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 2px;
    letter-spacing: 0.1em;
    margin-right: 6px;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Share Tech Mono', monospace;
    color: #00c8ff;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border-left: 3px solid #00c8ff;
    padding-left: 10px;
    margin: 28px 0 14px 0;
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 18px 0;
}
.metric-card {
    background: linear-gradient(135deg, rgba(5,20,40,0.95) 0%, rgba(2,11,24,0.95) 100%);
    border: 1px solid rgba(0,200,255,0.15);
    border-radius: 4px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent, #00c8ff), transparent);
}
.metric-card:hover { border-color: rgba(0,200,255,0.4); }
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: #6b8ba4;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Exo 2', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    line-height: 1;
    color: var(--accent, #00c8ff);
    text-shadow: 0 0 20px var(--accent, #00c8ff);
}
.metric-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: #3d5a72;
    margin-top: 6px;
    letter-spacing: 0.1em;
}

/* ── Packet Table ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,200,255,0.15) !important;
    border-radius: 4px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] thead tr th {
    background: rgba(0,200,255,0.06) !important;
    color: #00c8ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    border-bottom: 1px solid rgba(0,200,255,0.2) !important;
}
[data-testid="stDataFrame"] tbody tr {
    border-bottom: 1px solid rgba(0,200,255,0.05) !important;
}
[data-testid="stDataFrame"] tbody tr:hover td {
    background: rgba(0,200,255,0.04) !important;
}
[data-testid="stDataFrame"] tbody td {
    color: #a8c8d8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── Alert Boxes ── */
.alert-danger {
    background: linear-gradient(135deg, rgba(255,40,40,0.08), rgba(255,40,40,0.04));
    border: 1px solid rgba(255,60,60,0.4);
    border-left: 4px solid #ff3c3c;
    border-radius: 4px;
    padding: 16px 20px;
    color: #ff8080;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.05em;
    animation: pulse-red 2s infinite;
    margin: 14px 0;
}
.alert-safe {
    background: linear-gradient(135deg, rgba(0,255,150,0.06), rgba(0,255,150,0.02));
    border: 1px solid rgba(0,255,150,0.3);
    border-left: 4px solid #00ff96;
    border-radius: 4px;
    padding: 16px 20px;
    color: #00ff96;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.05em;
    margin: 14px 0;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,60,60,0); }
    50%       { box-shadow: 0 0 12px 2px rgba(255,60,60,0.25); }
}

/* ── Attack Type Pills ── */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 2px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    font-weight: 600;
}
.pill-normal  { background: rgba(0,255,150,0.12); color: #00ff96; border: 1px solid rgba(0,255,150,0.3); }
.pill-dos     { background: rgba(255,60,60,0.12);  color: #ff5555; border: 1px solid rgba(255,60,60,0.3); }
.pill-probe   { background: rgba(255,200,0,0.12);  color: #ffc800; border: 1px solid rgba(255,200,0,0.3); }
.pill-r2l     { background: rgba(255,100,0,0.12);  color: #ff6400; border: 1px solid rgba(255,100,0,0.3); }
.pill-u2r     { background: rgba(200,0,255,0.12);  color: #c800ff; border: 1px solid rgba(200,0,255,0.3); }

/* ── Model Info Tags ── */
.model-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,200,255,0.06);
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 2px;
    padding: 6px 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #5ab4d4;
    letter-spacing: 0.1em;
    margin-right: 8px;
    margin-bottom: 8px;
}
.model-dot {
    width: 6px; height: 6px;
    background: #00c8ff;
    border-radius: 50%;
    box-shadow: 0 0 6px #00c8ff;
    animation: blink 1.4s infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.2; }
}

/* ── Slider / Sidebar widgets ── */
[data-testid="stSlider"] .st-bf { background: #00c8ff !important; }
[data-testid="stSlider"] .st-bg { background: rgba(0,200,255,0.2) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #020b18; }
::-webkit-scrollbar-thumb { background: rgba(0,200,255,0.3); border-radius: 2px; }

/* ── Divider ── */
hr { border-color: rgba(0,200,255,0.1) !important; margin: 20px 0 !important; }

/* ── Status bar ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(0,200,255,0.04);
    border: 1px solid rgba(0,200,255,0.1);
    border-radius: 3px;
    padding: 8px 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: #4a7a9b;
    letter-spacing: 0.08em;
    margin-bottom: 24px;
}
.status-dot {
    width: 7px; height: 7px;
    background: #00ff96;
    border-radius: 50%;
    box-shadow: 0 0 8px #00ff96;
    animation: blink 1s infinite;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_models():
    X, y, scaler = load_and_preprocess("../data/network_data.csv")
    ml_model = joblib.load("../models/ml_model.pkl")
    nn_model = Net(X.shape[1])
    nn_model.load_state_dict(torch.load("../models/nn_model.pth", map_location="cpu"))
    nn_model.eval()
    return X, y, scaler, ml_model, nn_model

with st.spinner("Initializing CyberShield AI engine..."):
    X, y, scaler, ml_model, nn_model = load_models()


labels = {0: "dos", 1: "normal", 2: "probe", 3: "r2l", 4: "u2r"}

attack_colors = {
    "normal": "#00ff96",
    "dos":    "#ff5555",
    "probe":  "#ffc800",
    "r2l":    "#ff6400",
    "u2r":    "#c800ff",
}


with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
                    color:#00c8ff; letter-spacing:0.2em; margin-bottom:6px;">
            ◈ CYBERSHIELD AI ◈
        </div>
        <div style="font-size:2.8rem; filter:drop-shadow(0 0 14px #00c8ff);">🛡️</div>
        <div style="font-family:'Exo 2',sans-serif; font-weight:800; font-size:1.3rem;
                    color:#fff; letter-spacing:0.06em; margin-top:8px;">
            CONTROL PANEL
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-family:\'Share Tech Mono\',monospace; font-size:0.7rem; color:#00c8ff; letter-spacing:0.15em;">⚙ SCAN PARAMETERS</p>', unsafe_allow_html=True)
    num_packets = st.slider("Packet batch size", 1, 30, 8, help="Number of packets to analyze per scan")

    auto_refresh = st.checkbox("⟳  Auto-refresh (5s)", value=False)
    show_rf = st.checkbox("🌲  Show Random Forest column", value=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.62rem; color:#3d5a72; line-height:1.8; letter-spacing:0.08em;">
    ATTACK LEGEND<br>
    <span style="color:#ff5555">■</span> DOS &nbsp;— Denial of Service<br>
    <span style="color:#ffc800">■</span> PROBE — Network Scan<br>
    <span style="color:#ff6400">■</span> R2L &nbsp;— Remote to Local<br>
    <span style="color:#c800ff">■</span> U2R &nbsp;— User to Root<br>
    <span style="color:#00ff96">■</span> NORMAL — Safe Traffic
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄  Run New Scan", use_container_width=True, type="primary"):
        st.rerun()

st.markdown("""
<div class="cyber-header">
    <div class="cyber-header-icon">🛡️</div>
    <div>
        <div class="cyber-title">CYBERSHIELD AI</div>
        <div class="cyber-subtitle">HYBRID NETWORK INTRUSION DETECTION SYSTEM</div>
    </div>
</div>
""", unsafe_allow_html=True)

ts = time.strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(f"""
<div class="status-bar">
    <div class="status-dot"></div>
    SYSTEM ONLINE &nbsp;|&nbsp; TIMESTAMP: {ts} &nbsp;|&nbsp;
    MODELS: RANDOM FOREST + PYTORCH NN &nbsp;|&nbsp; FEATURES: {X.shape[1]}
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom:20px;">
    <span class="model-tag"><span class="model-dot"></span> 🌲 RANDOM FOREST</span>
    <span class="model-tag"><span class="model-dot"></span> 🧠 PYTORCH NEURAL NET</span>
    <span class="badge" style="display:inline-block; background:rgba(255,200,0,0.08); border:1px solid rgba(255,200,0,0.2);
        color:#ffc800; font-family:'Share Tech Mono',monospace; font-size:0.65rem; padding:5px 12px;
        border-radius:2px; letter-spacing:0.1em;">5-CLASS CLASSIFIER</span>
</div>
""", unsafe_allow_html=True)


packet_table = []
attack_type_counts = {k: 0 for k in labels.values()}

for _ in range(num_packets):
    sample = [
        random.randint(1, 100),
        random.randint(0, 2),
        random.randint(100, 10000),
        random.randint(0, 3000),
        random.randint(0, 2)
    ]
    columns = ['duration', 'protocol_type', 'src_bytes', 'dst_bytes', 'flag']
    sample_df = pd.DataFrame([sample], columns=columns)
    sample_scaled = scaler.transform(sample_df)

    rf_pred = ml_model.predict(sample_scaled)[0]
    rf_label = labels.get(rf_pred, str(rf_pred))

    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)
    with torch.no_grad():
        nn_pred = nn_model(sample_tensor)
    predicted_class = torch.argmax(nn_pred, dim=1).item()
    attack_type = labels[predicted_class]
    attack_type_counts[attack_type] += 1

    proto_map = {0: "TCP", 1: "UDP", 2: "ICMP"}
    flag_map  = {0: "SF", 1: "REJ", 2: "S0"}

    packet_table.append({
        "Duration":    sample[0],
        "Protocol":    proto_map.get(sample[1], str(sample[1])),
        "Src Bytes":   sample[2],
        "Dst Bytes":   sample[3],
        "Flag":        flag_map.get(sample[4], str(sample[4])),
        "RF Pred":     rf_label.upper(),
        "NN Pred":     attack_type.upper(),
        "Status":      "⚠ THREAT" if attack_type != "normal" else "✓ SAFE",
    })

attack_count = sum(v for k, v in attack_type_counts.items() if k != "normal")
normal_count = attack_type_counts["normal"]
threat_pct   = int(attack_count / num_packets * 100) if num_packets else 0


st.markdown('<div class="section-header">// SCAN SUMMARY</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card" style="--accent:#00c8ff;">
    <div class="metric-label">Packets Analyzed</div>
    <div class="metric-value">{num_packets}</div>
    <div class="metric-sub">THIS SCAN BATCH</div>
  </div>
  <div class="metric-card" style="--accent:#ff5555;">
    <div class="metric-label">Threats Detected</div>
    <div class="metric-value">{attack_count}</div>
    <div class="metric-sub">ATTACK PACKETS</div>
  </div>
  <div class="metric-card" style="--accent:#00ff96;">
    <div class="metric-label">Normal Traffic</div>
    <div class="metric-value">{normal_count}</div>
    <div class="metric-sub">SAFE PACKETS</div>
  </div>
  <div class="metric-card" style="--accent:{'#ff5555' if threat_pct > 40 else '#ffc800' if threat_pct > 0 else '#00ff96'};">
    <div class="metric-label">Threat Score</div>
    <div class="metric-value">{threat_pct}%</div>
    <div class="metric-sub">OF TOTAL TRAFFIC</div>
  </div>
</div>
""", unsafe_allow_html=True)


if attack_count > 0:
    top_threat = max((k for k in attack_type_counts if k != "normal"), key=lambda k: attack_type_counts[k])
    st.markdown(f"""
    <div class="alert-danger">
        ⚠&nbsp; INTRUSION ALERT — {attack_count} MALICIOUS PACKET(S) DETECTED &nbsp;|&nbsp;
        DOMINANT THREAT TYPE: <strong>{top_threat.upper()}</strong> ({attack_type_counts[top_threat]} packets)
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="alert-safe">
        ✓&nbsp; ALL CLEAR — No intrusion activity detected in this scan batch.
        Network traffic appears nominal.
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div class="section-header">// PACKET INSPECTOR</div>', unsafe_allow_html=True)

df = pd.DataFrame(packet_table)
if not show_rf:
    df = df.drop(columns=["RF Pred"])

st.dataframe(
    df,
    use_container_width=True,
    height=min(38 * len(df) + 42, 420),
    hide_index=True,
)


st.markdown('<div class="section-header">// TRAFFIC ANALYTICS</div>', unsafe_allow_html=True)

col_a, col_b = st.columns([3, 2], gap="large")

with col_a:
    st.markdown('<p style="font-family:\'Share Tech Mono\',monospace; font-size:0.68rem; color:#4a7a9b; letter-spacing:0.12em; margin-bottom:8px;">ATTACK TYPE DISTRIBUTION</p>', unsafe_allow_html=True)
    chart_df = pd.DataFrame({
        "Type":  [k.upper() for k in attack_type_counts],
        "Count": list(attack_type_counts.values())
    }).set_index("Type")
    st.bar_chart(chart_df, height=220, use_container_width=True)

with col_b:
    st.markdown('<p style="font-family:\'Share Tech Mono\',monospace; font-size:0.68rem; color:#4a7a9b; letter-spacing:0.12em; margin-bottom:8px;">THREAT BREAKDOWN</p>', unsafe_allow_html=True)
    for label, count in attack_type_counts.items():
        pct   = int(count / num_packets * 100) if num_packets else 0
        color = attack_colors.get(label, "#00c8ff")
        bar_w = max(pct, 2) if count > 0 else 0
        st.markdown(f"""
        <div style="margin-bottom:10px;">
          <div style="display:flex; justify-content:space-between; align-items:center;
                      font-family:'Share Tech Mono',monospace; font-size:0.68rem;
                      color:{color}; letter-spacing:0.1em; margin-bottom:3px;">
            <span>{label.upper()}</span>
            <span>{count} pkt &nbsp;{pct}%</span>
          </div>
          <div style="background:rgba(255,255,255,0.04); border-radius:1px; height:5px; overflow:hidden;">
            <div style="background:{color}; width:{bar_w}%; height:100%;
                        box-shadow:0 0 8px {color}; transition:width 0.5s;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center;
            font-family:'Share Tech Mono',monospace; font-size:0.6rem;
            color:#2a4a62; letter-spacing:0.1em; padding-bottom:12px;">
    <span>◈ CYBERSHIELD AI · HYBRID IDS v2.0</span>
    <span>RANDOM FOREST + PYTORCH NN · 5-CLASS CLASSIFICATION</span>
    <span>KDD CUP 99 SCHEMA</span>
</div>
""", unsafe_allow_html=True)


if auto_refresh:
    time.sleep(5)
    st.rerun()