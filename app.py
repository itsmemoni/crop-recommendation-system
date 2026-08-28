    import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ============================================================
# LOAD MODEL, SCALER, LABEL ENCODER
# ============================================================
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# ============================================================
# FERTILIZER RECOMMENDATION FUNCTION
# ============================================================
def fertilizer_recommend(N, P, K):
    if N < 50 and P < 50 and K < 50:
        return "DAP (Di-Ammonium Phosphate) — Low NPK Soil"
    elif N > 100:
        return "Urea — High Nitrogen Needed"
    elif P > 100:
        return "SSP (Single Super Phosphate)"
    elif K > 100:
        return "MOP (Muriate of Potash)"
    elif N < 50:
        return "Urea or Ammonium Sulphate — Low Nitrogen"
    elif P < 50:
        return "SSP or DAP — Low Phosphorus"
    elif K < 50:
        return "MOP or SOP — Low Potassium"
    else:
        return "NPK 17:17:17 — Balanced Fertilizer"

# ============================================================
# YIELD & MARKET PRICE DATA
# ============================================================
yield_data = {
    'rice': 3500, 'maize': 2800, 'chickpea': 1200,
    'kidneybeans': 1500, 'pigeonpeas': 900,
    'mothbeans': 800, 'mungbean': 1000,
    'blackgram': 950, 'lentil': 1100,
    'pomegranate': 8000, 'banana': 25000,
    'mango': 6000, 'grapes': 12000,
    'watermelon': 20000, 'muskmelon': 15000,
    'apple': 10000, 'orange': 9000,
    'papaya': 18000, 'coconut': 5500,
    'cotton': 1800, 'jute': 2200, 'coffee': 800
}

market_price = {
    'rice': 20, 'maize': 15, 'chickpea': 60,
    'kidneybeans': 80, 'pigeonpeas': 70,
    'mothbeans': 55, 'mungbean': 65,
    'blackgram': 60, 'lentil': 75,
    'pomegranate': 80, 'banana': 25, 'mango': 50,
    'grapes': 90, 'watermelon': 15, 'muskmelon': 20,
    'apple': 100, 'orange': 40, 'papaya': 20,
    'coconut': 30, 'cotton': 55, 'jute': 35,
    'coffee': 350
}

# Crop emoji map for visual flair
crop_emoji = {
    'rice': '🌾', 'maize': '🌽', 'chickpea': '🫘', 'kidneybeans': '🫘',
    'pigeonpeas': '🫛', 'mothbeans': '🫘', 'mungbean': '🫘', 'blackgram': '🫘',
    'lentil': '🫘', 'pomegranate': '🍑', 'banana': '🍌', 'mango': '🥭',
    'grapes': '🍇', 'watermelon': '🍉', 'muskmelon': '🍈', 'apple': '🍎',
    'orange': '🍊', 'papaya': '🥭', 'coconut': '🥥', 'cotton': '☁️',
    'jute': '🌿', 'coffee': '☕'
}

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="centered"
)

# ============================================================
# CUSTOM CSS — NATURE / AGRICULTURE THEME
# ============================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Karla:wght@400;500;700&display=swap');

    :root {
        --soil: #3E2723;
        --leaf: #4A7C59;
        --leaf-dark: #345940;
        --harvest: #D4A24C;
        --cream: #F7F3E9;
        --sky: #7BA8B8;
        --card: #FFFFFF;
    }

    html, body, [class*="css"] {
        font-family: 'Karla', sans-serif;
    }

    .stApp {
        background:
            linear-gradient(180deg, rgba(247,243,233,0.97) 0%, rgba(240,234,215,0.97) 100%);
    }

    /* Field-row texture strip at the very top */
    .field-strip {
        height: 10px;
        width: 100%;
        background: repeating-linear-gradient(
            90deg,
            var(--leaf) 0px, var(--leaf) 28px,
            var(--harvest) 28px, var(--harvest) 36px
        );
        border-radius: 6px;
        margin-bottom: 1.6rem;
        opacity: 0.85;
    }

    .hero-wrap {
        text-align: center;
        padding: 0.4rem 0 1.2rem 0;
    }

    .hero-eyebrow {
        font-family: 'Karla', sans-serif;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.72rem;
        color: var(--leaf-dark);
        opacity: 0.85;
        margin-bottom: 0.4rem;
    }

    .hero-title {
        font-family: 'Fraunces', serif;
        font-weight: 700;
        font-size: 2.5rem;
        line-height: 1.08;
        color: var(--soil);
        margin: 0.1rem 0 0.5rem 0;
    }

    .hero-title span {
        color: var(--leaf);
        font-style: italic;
    }

    .hero-sub {
        font-family: 'Karla', sans-serif;
        font-size: 0.95rem;
        color: #6b5d4f;
        margin-bottom: 0.15rem;
    }

    .hero-team {
        font-size: 0.82rem;
        color: #9a8b78;
    }

    .section-card {
        background: var(--card);
        border: 1px solid rgba(74,124,89,0.18);
        border-radius: 16px;
        padding: 1.6rem 1.6rem 1.2rem 1.6rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 18px rgba(62,39,35,0.06);
    }

    .section-label {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.25rem;
        color: var(--soil);
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-caption {
        font-size: 0.82rem;
        color: #9a8b78;
        margin-bottom: 1.1rem;
    }

    /* Streamlit number input styling */
    div[data-testid="stNumberInput"] label {
        font-family: 'Karla', sans-serif;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: var(--leaf-dark);
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 10px !important;
        border: 1.5px solid rgba(74,124,89,0.25) !important;
        background: var(--cream) !important;
        font-family: 'Karla', sans-serif;
        font-weight: 500;
    }

    /* Primary button */
    .stButton > button {
        background: linear-gradient(135deg, var(--leaf) 0%, var(--leaf-dark) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 999px !important;
        padding: 0.85rem 1.5rem !important;
        font-family: 'Fraunces', serif !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.01em;
        box-shadow: 0 6px 20px rgba(74,124,89,0.35) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 26px rgba(74,124,89,0.45) !important;
    }

    /* Result hero card */
    .result-hero {
        background: linear-gradient(135deg, var(--leaf) 0%, var(--leaf-dark) 100%);
        border-radius: 20px;
        padding: 2rem 1.6rem;
        text-align: center;
        margin: 0.4rem 0 1.4rem 0;
        box-shadow: 0 10px 30px rgba(74,124,89,0.3);
    }

    .result-emoji {
        font-size: 3rem;
        margin-bottom: 0.2rem;
    }

    .result-label {
        font-family: 'Karla', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.8);
        margin-bottom: 0.3rem;
    }

    .result-crop {
        font-family: 'Fraunces', serif;
        font-weight: 700;
        font-size: 2.4rem;
        color: white;
        text-transform: capitalize;
    }

    /* Metric tiles */
    .metric-tile {
        background: var(--cream);
        border-left: 4px solid var(--harvest);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.9rem;
    }

    .metric-tile.blue { border-left-color: var(--sky); }
    .metric-tile.green { border-left-color: var(--leaf); }

    .metric-icon {
        font-size: 1.3rem;
        margin-bottom: 0.25rem;
        display: block;
    }

    .metric-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9a8b78;
        margin-bottom: 0.15rem;
    }

    .metric-value {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.15rem;
        color: var(--soil);
    }

    .footer-note {
        text-align: center;
        font-size: 0.78rem;
        color: #9a8b78;
        padding-top: 1rem;
        border-top: 1px solid rgba(74,124,89,0.15);
        margin-top: 1.5rem;
    }

    .footer-note b { color: var(--leaf-dark); }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# HERO SECTION
# ============================================================
st.markdown('<div class="field-strip"></div>', unsafe_allow_html=True)

st.markdown("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">Pavai Arts and Science College for Women</div>
        <div class="hero-title">Smart Crop <span>Recommendation</span></div>
        <div class="hero-sub">Soil-first guidance for what to grow, when, and how it pays off</div>
        <div class="hero-team">BSc AI &amp; DS · Team M.S. Monica &amp; A. Monisha</div>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# INPUT SECTION
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("""
    <div class="section-label">🌱 Field Readings</div>
    <div class="section-caption">Enter your soil test values and today's weather</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen · N", 0, 200, 90)
    P = st.number_input("Phosphorus · P", 0, 200, 42)
    K = st.number_input("Potassium · K", 0, 200, 43)
    temperature = st.number_input("Temperature °C", 0.0, 50.0, 20.8)

with col2:
    humidity = st.number_input("Humidity %", 0.0, 100.0, 82.0)
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall mm", 0.0, 500.0, 202.9)

st.markdown('</div>', unsafe_allow_html=True)

# Center the button
bcol1, bcol2, bcol3 = st.columns([1, 2, 1])
with bcol2:
    predict_clicked = st.button("🌾  Recommend My Crop", use_container_width=True)

# ============================================================
# PREDICTION & RESULTS
# ============================================================
if predict_clicked:

    sample = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

    sample_scaled = scaler.transform(sample)
    pred_encoded = model.predict(sample_scaled)
    crop = le.inverse_transform(pred_encoded)[0]

    fertilizer = fertilizer_recommend(N, P, K)
    expected_yield = yield_data.get(crop, 1000)
    price = market_price.get(crop, 30)
    profit = expected_yield * price
    emoji = crop_emoji.get(crop, '🌱')

    st.markdown(f"""
        <div class="result-hero">
            <div class="result-emoji">{emoji}</div>
            <div class="result-label">Best Match For Your Field</div>
            <div class="result-crop">{crop}</div>
        </div>
    """, unsafe_allow_html=True)

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.markdown(f"""
            <div class="metric-tile">
                <span class="metric-icon">💊</span>
                <div class="metric-label">Fertilizer</div>
                <div class="metric-value">{fertilizer}</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="metric-tile green">
                <span class="metric-icon">📈</span>
                <div class="metric-label">Expected Yield</div>
                <div class="metric-value">{expected_yield:,} kg/hectare</div>
            </div>
        """, unsafe_allow_html=True)
    with mcol2:
        st.markdown(f"""
            <div class="metric-tile blue">
                <span class="metric-icon">💰</span>
                <div class="metric-label">Market Price</div>
                <div class="metric-value">₹{price}/kg</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="metric-tile">
                <span class="metric-icon">💵</span>
                <div class="metric-label">Expected Profit</div>
                <div class="metric-value">₹{profit:,}</div>
            </div>
        """, unsafe_allow_html=True)

    st.balloons()

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
    <div class="footer-note">
        Guide: <b>Ms. Anithakumari</b> &nbsp;·&nbsp; BSc Artificial Intelligence &amp; Data Science &nbsp;·&nbsp; 5th Semester Project
    </div>
""", unsafe_allow_html=True)
