"""
Titanic Survival Prediction System
Streamlit Application — ANN Model Deployment
Corrected & redesigned version
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ── Optional TensorFlow import (graceful fallback if not installed) ──────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ============================================================
# PAGE CONFIGURATION  (must be the very first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0a1628 0%, #0f2545 55%, #16345e 100%);
    padding: 3rem 2rem 2.5rem;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        90deg, transparent, transparent 40px,
        rgba(255,255,255,0.018) 40px, rgba(255,255,255,0.018) 41px
    ), repeating-linear-gradient(
        0deg, transparent, transparent 40px,
        rgba(255,255,255,0.018) 40px, rgba(255,255,255,0.018) 41px
    );
    pointer-events: none;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    color: #ffffff;
    margin-bottom: 0.4rem;
}
.hero-sub {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.45);
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.hero-badges { margin-top: 1.2rem; display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
.badge {
    font-size: 0.7rem;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.18);
    color: rgba(255,255,255,0.6);
    background: rgba(255,255,255,0.07);
    letter-spacing: 0.05em;
    display: inline-block;
}

/* ── Panels ── */
.panel {
    background: #ffffff;
    border: 1px solid #e8ecf0;
    border-radius: 14px;
    padding: 1.8rem 1.6rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.panel-label {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 1.2rem;
    font-weight: 500;
}

/* ── Verdict cards ── */
.verdict-survived {
    background: #ecfdf5;
    border: 1px solid #6ee7b7;
    border-radius: 12px;
    padding: 1.8rem 1rem;
    text-align: center;
}
.verdict-perished {
    background: #fef2f2;
    border: 1px solid #fca5a5;
    border-radius: 12px;
    padding: 1.8rem 1rem;
    text-align: center;
}
.verdict-word-s { font-family: 'Playfair Display', serif; font-size: 1.9rem; color: #059669; }
.verdict-word-p { font-family: 'Playfair Display', serif; font-size: 1.9rem; color: #dc2626; }
.verdict-detail { font-size: 0.82rem; color: #6b7280; margin-top: 4px; }

/* ── Metric tiles ── */
.metric-row { display: flex; gap: 10px; margin: 1rem 0; }
.metric-tile {
    flex: 1;
    background: #f8fafc;
    border: 1px solid #e8ecf0;
    border-radius: 10px;
    padding: 0.9rem;
    text-align: center;
}
.mt-val { font-size: 1.3rem; font-weight: 500; color: #111827; display: block; }
.mt-lbl { font-size: 0.65rem; letter-spacing: 0.08em; text-transform: uppercase; color: #9ca3af; display: block; margin-top: 2px; }

/* ── Probability bars ── */
.prob-section { margin-top: 1rem; }
.prob-lbl { font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; color: #9ca3af; margin-bottom: 8px; }
.bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.bar-name { font-size: 0.78rem; color: #6b7280; width: 72px; }
.bar-track { flex: 1; height: 10px; background: #f1f5f9; border-radius: 10px; overflow: hidden; }
.bar-fill-s { height: 100%; border-radius: 10px; background: #10b981; }
.bar-fill-p { height: 100%; border-radius: 10px; background: #ef4444; }
.bar-pct { font-size: 0.78rem; font-weight: 500; width: 40px; text-align: right; color: #374151; }

/* ── Info footer ── */
.info-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 2rem; }
.info-tile {
    background: #f8fafc;
    border: 1px solid #e8ecf0;
    border-radius: 10px;
    padding: 1rem;
}
.info-title { font-size: 0.82rem; font-weight: 500; color: #374151; margin-bottom: 4px; }
.info-body { font-size: 0.75rem; color: #6b7280; line-height: 1.55; }

/* ── Streamlit element overrides ── */
div[data-testid="stForm"] { border: none; padding: 0; }
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    font-size: 0.82rem !important;
    color: #374151 !important;
    font-weight: 500 !important;
}
div[data-testid="stFormSubmitButton"] button {
    background: #0a1628 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1rem !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stFormSubmitButton"] button:hover { opacity: 0.85 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO BANNER
# ============================================================
st.markdown("""
<div class="hero">
    <div style="font-size:2.4rem; margin-bottom:0.4rem;">🚢</div>
    <div class="hero-title">Titanic Survival Predictor</div>
    <div class="hero-sub">Artificial Neural Network &nbsp;·&nbsp; April 15, 1912</div>
    <div class="hero-badges">
        <span class="badge">ANN Architecture 3→2→1</span>
        <span class="badge">Min-Max Normalized</span>
        <span class="badge">Sigmoid Activation</span>
        <span class="badge">TensorFlow / Fallback Mode</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MODEL LOADING  (with safe fallback)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load trained TF model if it exists; otherwise return None."""
    if not TF_AVAILABLE:
        return None
    model_path = Path("titanic_model.h5")
    if model_path.exists():
        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception:
            return None
    return None

model = load_model()

# ============================================================
# PREPROCESSING UTILITIES
# ============================================================
def preprocess_input(pclass: int, age: float, fare: float) -> np.ndarray:
    """Min-Max scale inputs to [0, 1] range."""
    pclass_norm = (pclass - 1) / 2          # [1, 3] → [0, 1]
    age_norm    = np.clip(age, 0, 80) / 80  # [0, 80] → [0, 1]
    fare_norm   = np.clip(fare, 0, 500) / 500  # [0, 500] → [0, 1]
    return np.array([[pclass_norm, age_norm, fare_norm]], dtype=np.float32)


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    return float(1 / (1 + np.exp(-np.clip(x, -500, 500))))


def manual_forward_pass(pclass: int, age: float, fare: float) -> float:
    """
    Demonstration ANN (3-2-1) forward pass with domain-aware bias correction.
    Replace weights/biases with your actual trained values.
    """
    x = preprocess_input(pclass, age, fare)[0]

    # ── Hidden layer weights (2 neurons × 3 inputs) ──────────────────────
    W_ih = np.array([[0.11, 0.14, 0.17],
                     [0.21, 0.24, 0.27]], dtype=np.float32)
    b_h  = np.array([0.10, 0.10], dtype=np.float32)

    # ── Output layer weights (1 neuron × 2 hidden) ───────────────────────
    W_ho = np.array([0.31, 0.34], dtype=np.float32)
    b_o  = 0.10

    # ── Forward propagation ───────────────────────────────────────────────
    z_h = W_ih @ x + b_h          # shape (2,)
    a_h = 1 / (1 + np.exp(-z_h))  # sigmoid, shape (2,)
    z_o = float(W_ho @ a_h) + b_o
    raw = sigmoid(z_o)

    # ── Domain-aware bias corrections (reflect real Titanic statistics) ──
    # Higher class → better survival odds
    class_boost  = (3 - pclass) * 0.12
    # Higher fare  → better survival odds (correlated with class/cabin)
    fare_boost   = (fare / 500) * 0.18
    # Age penalty: very old passengers had worse survival rates
    age_penalty  = 0.06 if age > 60 else (-0.08 if age < 12 else 0.0)

    adjusted = raw + class_boost + fare_boost - age_penalty
    return float(np.clip(adjusted, 0.02, 0.97))


def predict(pclass: int, age: float, fare: float) -> float:
    """
    Run inference: real model if loaded, otherwise manual ANN fallback.
    Returns survival probability in [0, 1].
    """
    if model is not None:
        input_data = preprocess_input(pclass, age, fare)
        prob = float(model.predict(input_data, verbose=0)[0][0])
        return float(np.clip(prob, 0.0, 1.0))
    return manual_forward_pass(pclass, age, fare)

# ============================================================
# LAYOUT — two-column grid
# ============================================================
col_input, col_output = st.columns([1, 1], gap="large")

# ── LEFT PANEL: Passenger Input ──────────────────────────────────────────────
with col_input:
    st.markdown('<p class="panel-label">Passenger details</p>', unsafe_allow_html=True)

    with st.form("prediction_form", clear_on_submit=False):
        pclass = st.selectbox(
            "🎫 Passenger class",
            options=[1, 2, 3],
            format_func=lambda x: f"Class {x} — {'First' if x==1 else 'Second' if x==2 else 'Third'}",
            help="Ticket class: 1 = First Class, 2 = Second Class, 3 = Third Class",
        )
        age = st.slider(
            "🎂 Age (years)",
            min_value=1,
            max_value=80,
            value=28,
            step=1,
            help="Passenger age in years",
        )
        fare = st.number_input(
            "💰 Ticket fare (£)",
            min_value=0.0,
            max_value=500.0,
            value=32.0,
            step=1.0,
            format="%.2f",
            help="Ticket fare paid in British Pounds",
        )
        submitted = st.form_submit_button(
            "🔮  Predict survival",
            use_container_width=True,
            type="primary",
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Model status indicator
    status = "✅ TF model loaded" if model is not None else "⚙️ Demo ANN (no .h5 found)"
    st.caption(f"Model status: {status}")

# ── RIGHT PANEL: Prediction Output ──────────────────────────────────────────
with col_output:
    st.markdown('<p class="panel-label">Prediction output</p>', unsafe_allow_html=True)

    if not submitted:
        st.info("👈 Fill in the passenger details and click **Predict survival** to see results.")

        # Placeholder empty gauge
        fig_placeholder = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Waiting for input…", "font": {"size": 16, "color": "#9ca3af"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#e5e7eb"},
                "bar": {"color": "#e5e7eb"},
                "bgcolor": "#f9fafb",
                "bordercolor": "#e5e7eb",
            },
            number={"font": {"color": "#d1d5db"}},
        ))
        fig_placeholder.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20),
                                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_placeholder, use_container_width=True)

    else:
        # ── Run inference ────────────────────────────────────────────────
        prob      = predict(pclass, age, fare)
        survived  = prob >= 0.5
        confidence = prob if survived else (1.0 - prob)

        # ── Verdict card ─────────────────────────────────────────────────
        if survived:
            st.markdown(f"""
            <div class="verdict-survived">
                <div style="font-size:2rem; margin-bottom:6px;">✅</div>
                <div class="verdict-word-s">Survived</div>
                <div class="verdict-detail">Model predicts this passenger likely survived</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-perished">
                <div style="font-size:2rem; margin-bottom:6px;">✖️</div>
                <div class="verdict-word-p">Perished</div>
                <div class="verdict-detail">Model predicts this passenger did not survive</div>
            </div>""", unsafe_allow_html=True)

        # ── Metric tiles ─────────────────────────────────────────────────
        risk = "Low" if prob > 0.7 else "Medium" if prob > 0.4 else "High"
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-tile">
                <span class="mt-val">{prob:.1%}</span>
                <span class="mt-lbl">Survival prob.</span>
            </div>
            <div class="metric-tile">
                <span class="mt-val">{confidence:.1%}</span>
                <span class="mt-lbl">Confidence</span>
            </div>
            <div class="metric-tile">
                <span class="mt-val">{risk}</span>
                <span class="mt-lbl">Risk level</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Probability bars ──────────────────────────────────────────────
        st.markdown(f"""
        <div class="prob-section">
            <div class="prob-lbl">Probability breakdown</div>
            <div class="bar-row">
                <span class="bar-name">Survived</span>
                <div class="bar-track"><div class="bar-fill-s" style="width:{prob:.1%}"></div></div>
                <span class="bar-pct">{prob:.1%}</span>
            </div>
            <div class="bar-row">
                <span class="bar-name">Perished</span>
                <div class="bar-track"><div class="bar-fill-p" style="width:{1-prob:.1%}"></div></div>
                <span class="bar-pct">{1-prob:.1%}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Gauge chart ───────────────────────────────────────────────────
        bar_color = "#10b981" if survived else "#ef4444"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(prob * 100, 1),
            delta={"reference": 50, "increasing": {"color": "#10b981"}, "decreasing": {"color": "#ef4444"}},
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Survival probability (%)", "font": {"size": 15, "color": "#6b7280"}},
            number={"font": {"size": 36, "color": bar_color}, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#e5e7eb"},
                "bar": {"color": bar_color},
                "bgcolor": "#f9fafb",
                "borderwidth": 1,
                "bordercolor": "#e5e7eb",
                "steps": [
                    {"range": [0, 50],  "color": "#fef2f2"},
                    {"range": [50, 100], "color": "#ecfdf5"},
                ],
                "threshold": {
                    "line": {"color": "#374151", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Donut chart ───────────────────────────────────────────────────
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Survived", "Perished"],
            values=[round(prob, 4), round(1 - prob, 4)],
            hole=0.55,
            marker_colors=["#10b981", "#ef4444"],
            textinfo="label+percent",
            textfont_size=13,
            pull=[0.04, 0],
        )])
        fig_donut.update_layout(
            title=dict(text="Survival distribution", font=dict(size=14, color="#6b7280")),
            height=300,
            margin=dict(l=20, r=20, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            annotations=[dict(text=f"{prob:.0%}", x=0.5, y=0.5,
                              font=dict(size=22, color=bar_color), showarrow=False)],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# INFO FOOTER
# ============================================================
st.markdown("""
<div class="info-grid">
    <div class="info-tile">
        <div class="info-title">🧠 Model architecture</div>
        <div class="info-body">3 input neurons → 2 hidden (sigmoid) → 1 output (sigmoid).
        Forward propagation with min-max normalization on all features.</div>
    </div>
    <div class="info-tile">
        <div class="info-title">📊 Features used</div>
        <div class="info-body">Passenger class [1–3], age [1–80 years], and ticket fare [£0–500],
        each normalized to a [0, 1] range before inference.</div>
    </div>
    <div class="info-tile">
        <div class="info-title">📜 About the data</div>
        <div class="info-body">RMS Titanic sank April 15 1912. 1,517 of 2,224 passengers perished.
        Class and fare are strongly correlated with survival rate.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#9ca3af; font-size:0.75rem; padding:1rem 0;">
    🚢 <strong>Titanic Survival Prediction System</strong>
    &nbsp;·&nbsp; TensorFlow &amp; Streamlit
    &nbsp;·&nbsp; ANN Architecture 3-2-1
</div>
""", unsafe_allow_html=True)