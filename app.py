import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests

# ============================================================
# LOAD MODEL, SCALER, LABEL ENCODER
# ============================================================
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# ============================================================
# WEATHER API CONFIG (OpenWeatherMap)
# ============================================================
# Get a free API key at https://openweathermap.org/api
# Paste it below between the quotes.
WEATHER_API_KEY = "01aa1eccb6b748ad636894b83fbd352c"

def fetch_weather(city):
    """Fetch live temperature, humidity and a rainfall estimate for a city."""
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={WEATHER_API_KEY}&units=metric"
        )
        response = requests.get(url, timeout=6)
        data = response.json()

        if response.status_code != 200 or "main" not in data:
            return None

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        # OpenWeatherMap only gives rain volume for the last 1-3 hours if it's
        # currently raining; otherwise we fall back to a light default.
        rainfall = data.get("rain", {}).get("1h", 0) * 24  # rough daily estimate
        if rainfall == 0:
            rainfall = 50.0  # sensible fallback when it isn't raining right now

        return {
            "temperature": round(temperature, 1),
            "humidity": round(humidity, 1),
            "rainfall": round(rainfall, 1),
            "description": data["weather"][0]["description"].title()
        }
    except Exception:
        return None

# ============================================================
# LANGUAGE STRINGS (English / Tamil)
# ============================================================
TEXT = {
    "en": {
        "title": "Smart Crop Recommendation",
        "subtitle": "Soil-first guidance for what to grow, when, and how it pays off",
        "college": "Pavai Arts and Science College for Women",
        "team": "BSc AI & DS · Team M.S. Monica & A. Monisha",
        "weather_header": "Fetch Live Weather",
        "weather_caption": "Enter your city to auto-fill temperature, humidity and rainfall",
        "city_label": "City",
        "fetch_button": "Fetch Weather",
        "weather_success": "Weather fetched for",
        "weather_fail": "Could not fetch weather for this city. Please check the spelling or enter values manually.",
        "field_header": "Field Readings",
        "field_caption": "Enter your soil test values and today's weather",
        "n_label": "Nitrogen · N",
        "p_label": "Phosphorus · P",
        "k_label": "Potassium · K",
        "temp_label": "Temperature °C",
        "humidity_label": "Humidity %",
        "ph_label": "Soil pH",
        "rainfall_label": "Rainfall mm",
        "predict_button": "Recommend My Crop",
        "results_label": "Best Match For Your Field",
        "fertilizer_label": "Fertilizer",
        "yield_label": "Expected Yield",
        "price_label": "Market Price",
        "profit_label": "Expected Profit",
        "footer": "Guide: Ms. Anithakumari · BSc Artificial Intelligence & Data Science · 5th Semester Project",
        "kg_hectare": "kg/hectare"
    },
    "ta": {
        "title": "ஸ்மார்ட் பயிர் பரிந்துரை",
        "subtitle": "உங்கள் மண்ணுக்கு எந்த பயிர் ஏற்றது, எப்போது, எவ்வளவு லாபம் என்பதை அறியுங்கள்",
        "college": "பாவை கலை அறிவியல் மகளிர் கல்லூரி",
        "team": "BSc AI & DS · குழு: எம்.எஸ். மோனிகா & ஏ. மோனிஷா",
        "weather_header": "நேரடி வானிலை பெறவும்",
        "weather_caption": "வெப்பநிலை, ஈரப்பதம், மழையளவு தானாக நிரப்ப உங்கள் ஊரை உள்ளிடவும்",
        "city_label": "ஊர்",
        "fetch_button": "வானிலை பெறவும்",
        "weather_success": "வானிலை பெறப்பட்டது",
        "weather_fail": "இந்த ஊருக்கான வானிலை கிடைக்கவில்லை. பெயரை சரிபார்க்கவும் அல்லது கைமுறையாக உள்ளிடவும்.",
        "field_header": "வயல் விவரங்கள்",
        "field_caption": "உங்கள் மண் பரிசோதனை மதிப்புகளையும் இன்றைய வானிலையையும் உள்ளிடவும்",
        "n_label": "நைட்ரஜன் · N",
        "p_label": "பாஸ்பரஸ் · P",
        "k_label": "பொட்டாசியம் · K",
        "temp_label": "வெப்பநிலை °C",
        "humidity_label": "ஈரப்பதம் %",
        "ph_label": "மண் pH",
        "rainfall_label": "மழையளவு mm",
        "predict_button": "பயிரை பரிந்துரை செய்",
        "results_label": "உங்கள் வயலுக்கு ஏற்ற பயிர்",
        "fertilizer_label": "உரம்",
        "yield_label": "எதிர்பார்க்கும் விளைச்சல்",
        "price_label": "சந்தை விலை",
        "profit_label": "எதிர்பார்க்கும் லாபம்",
        "footer": "வழிகாட்டி: திருமதி. அனிதகுமாரி · BSc செயற்கை நுண்ணறிவு மற்றும் தரவு அறிவியல் · 5வது செமஸ்டர் திட்டம்",
        "kg_hectare": "கிலோ/ஹெக்டேர்"
    }
}

# Crop names in Tamil for a friendlier result display
CROP_NAME_TA = {
    'rice': 'நெல்', 'maize': 'மக்காச்சோளம்', 'chickpea': 'கொண்டைக்கடலை',
    'kidneybeans': 'ராஜ்மா', 'pigeonpeas': 'துவரை', 'mothbeans': 'மோத் பீன்ஸ்',
    'mungbean': 'பாசிப்பயறு', 'blackgram': 'உளுந்து', 'lentil': 'பருப்பு',
    'pomegranate': 'மாதுளை', 'banana': 'வாழை', 'mango': 'மாம்பழம்',
    'grapes': 'திராட்சை', 'watermelon': 'தர்பூசணி', 'muskmelon': 'முலாம்பழம்',
    'apple': 'ஆப்பிள்', 'orange': 'ஆரஞ்சு', 'papaya': 'பப்பாளி',
    'coconut': 'தேங்காய்', 'cotton': 'பருத்தி', 'jute': 'சணல்', 'coffee': 'காபி'
}

# ============================================================
# FERTILIZER RECOMMENDATION FUNCTION
# ============================================================
def fertilizer_recommend(N, P, K, lang):
    if lang == "ta":
        if N < 50 and P < 50 and K < 50:
            return "டிஏபி (DAP) — குறைந்த NPK மண்"
        elif N > 100:
            return "யூரியா — அதிக நைட்ரஜன் தேவை"
        elif P > 100:
            return "எஸ்எஸ்பி (SSP)"
        elif K > 100:
            return "எம்ஓபி (MOP)"
        elif N < 50:
            return "யூரியா அல்லது அமோனியம் சல்பேட் — குறைந்த நைட்ரஜன்"
        elif P < 50:
            return "எஸ்எஸ்பி அல்லது டிஏபி — குறைந்த பாஸ்பரஸ்"
        elif K < 50:
            return "எம்ஓபி அல்லது எஸ்ஓபி — குறைந்த பொட்டாசியம்"
        else:
            return "என்பிகே 17:17:17 — சமச்சீர் உரம்"
    else:
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
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Karla:wght@400;500;700&family=Noto+Sans+Tamil:wght@400;600;700&display=swap');

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
        font-family: 'Karla', 'Noto Sans Tamil', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, rgba(247,243,233,0.97) 0%, rgba(240,234,215,0.97) 100%);
    }

    .field-strip {
        height: 10px;
        width: 100%;
        background: repeating-linear-gradient(
            90deg,
            var(--leaf) 0px, var(--leaf) 28px,
            var(--harvest) 28px, var(--harvest) 36px
        );
        border-radius: 6px;
        margin-bottom: 1rem;
        opacity: 0.85;
    }

    .lang-row {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 0.6rem;
    }

    .hero-wrap {
        text-align: center;
        padding: 0.4rem 0 1.2rem 0;
    }

    .hero-eyebrow {
        font-family: 'Karla', 'Noto Sans Tamil', sans-serif;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.72rem;
        color: var(--leaf-dark);
        opacity: 0.85;
        margin-bottom: 0.4rem;
    }

    .hero-title {
        font-family: 'Fraunces', 'Noto Sans Tamil', serif;
        font-weight: 700;
        font-size: 2.5rem;
        line-height: 1.15;
        color: var(--soil);
        margin: 0.1rem 0 0.5rem 0;
    }

    .hero-title span {
        color: var(--leaf);
        font-style: italic;
    }

    .hero-sub {
        font-family: 'Karla', 'Noto Sans Tamil', sans-serif;
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
        font-family: 'Fraunces', 'Noto Sans Tamil', serif;
        font-weight: 600;
        font-size: 1.25rem;
        color: var(--soil);
        margin-bottom: 0.2rem;
    }

    .section-caption {
        font-size: 0.82rem;
        color: #9a8b78;
        margin-bottom: 1.1rem;
    }

    div[data-testid="stNumberInput"] label, div[data-testid="stTextInput"] label {
        font-family: 'Karla', 'Noto Sans Tamil', sans-serif;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: var(--leaf-dark);
    }

    div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input {
        border-radius: 10px !important;
        border: 1.5px solid rgba(74,124,89,0.25) !important;
        background: var(--cream) !important;
        font-family: 'Karla', 'Noto Sans Tamil', sans-serif;
        font-weight: 500;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--leaf) 0%, var(--leaf-dark) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 999px !important;
        padding: 0.75rem 1.4rem !important;
        font-family: 'Fraunces', 'Noto Sans Tamil', serif !important;
        font-weight: 600 !important;
        font-size: 1.0rem !important;
        box-shadow: 0 6px 20px rgba(74,124,89,0.35) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 26px rgba(74,124,89,0.45) !important;
    }

    .weather-badge {
        background: linear-gradient(135deg, var(--sky) 0%, #5f8fa0 100%);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        color: white;
        font-size: 0.85rem;
        margin-top: 0.8rem;
    }

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
        font-family: 'Karla', 'Noto Sans Tamil', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.8);
        margin-bottom: 0.3rem;
    }

    .result-crop {
        font-family: 'Fraunces', 'Noto Sans Tamil', serif;
        font-weight: 700;
        font-size: 2.2rem;
        color: white;
        text-transform: capitalize;
    }

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
        font-family: 'Fraunces', 'Noto Sans Tamil', serif;
        font-weight: 600;
        font-size: 1.1rem;
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
# LANGUAGE SELECTOR
# ============================================================
if "lang" not in st.session_state:
    st.session_state.lang = "en"

st.markdown('<div class="lang-row">', unsafe_allow_html=True)
lang_choice = st.radio(
    "Language / மொழி",
    options=["English", "தமிழ்"],
    horizontal=True,
    label_visibility="collapsed"
)
st.session_state.lang = "en" if lang_choice == "English" else "ta"
st.markdown('</div>', unsafe_allow_html=True)

L = TEXT[st.session_state.lang]

# ============================================================
# HERO SECTION
# ============================================================
st.markdown('<div class="field-strip"></div>', unsafe_allow_html=True)

st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-eyebrow">{L['college']}</div>
        <div class="hero-title">{L['title']}</div>
        <div class="hero-sub">{L['subtitle']}</div>
        <div class="hero-team">{L['team']}</div>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# LIVE WEATHER SECTION
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(f"""
    <div class="section-label">🌦️ {L['weather_header']}</div>
    <div class="section-caption">{L['weather_caption']}</div>
""", unsafe_allow_html=True)

wcol1, wcol2 = st.columns([3, 1])
with wcol1:
    city = st.text_input(L['city_label'], value="Namakkal", label_visibility="collapsed", placeholder=L['city_label'])
with wcol2:
    fetch_clicked = st.button(L['fetch_button'], use_container_width=True)

if "weather" not in st.session_state:
    st.session_state.weather = None

if fetch_clicked:
    if WEATHER_API_KEY == "PASTE_YOUR_OPENWEATHERMAP_API_KEY_HERE":
        st.warning("⚠️ Add your free OpenWeatherMap API key in app.py (WEATHER_API_KEY) to enable this feature.")
    else:
        weather = fetch_weather(city)
        if weather:
            st.session_state.weather = weather
            st.markdown(f"""
                <div class="weather-badge">
                    ✅ {L['weather_success']} {city}: {weather['description']} ·
                    {weather['temperature']}°C · {weather['humidity']}% humidity
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error(L['weather_fail'])

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# INPUT SECTION
# ============================================================
w = st.session_state.weather

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(f"""
    <div class="section-label">🌱 {L['field_header']}</div>
    <div class="section-caption">{L['field_caption']}</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    N = st.number_input(L['n_label'], 0, 200, 90)
    P = st.number_input(L['p_label'], 0, 200, 42)
    K = st.number_input(L['k_label'], 0, 200, 43)
    temperature = st.number_input(
        L['temp_label'], 0.0, 50.0,
        float(w['temperature']) if w else 20.8
    )

with col2:
    humidity = st.number_input(
        L['humidity_label'], 0.0, 100.0,
        float(w['humidity']) if w else 82.0
    )
    ph = st.number_input(L['ph_label'], 0.0, 14.0, 6.5)
    rainfall = st.number_input(
        L['rainfall_label'], 0.0, 500.0,
        float(w['rainfall']) if w else 202.9
    )

st.markdown('</div>', unsafe_allow_html=True)

bcol1, bcol2, bcol3 = st.columns([1, 2, 1])
with bcol2:
    predict_clicked = st.button(f"🌾  {L['predict_button']}", use_container_width=True)

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

    fertilizer = fertilizer_recommend(N, P, K, st.session_state.lang)
    expected_yield = yield_data.get(crop, 1000)
    price = market_price.get(crop, 30)
    profit = expected_yield * price
    emoji = crop_emoji.get(crop, '🌱')

    display_crop = CROP_NAME_TA.get(crop, crop) if st.session_state.lang == "ta" else crop

    st.markdown(f"""
        <div class="result-hero">
            <div class="result-emoji">{emoji}</div>
            <div class="result-label">{L['results_label']}</div>
            <div class="result-crop">{display_crop}</div>
        </div>
    """, unsafe_allow_html=True)

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.markdown(f"""
            <div class="metric-tile">
                <span class="metric-icon">💊</span>
                <div class="metric-label">{L['fertilizer_label']}</div>
                <div class="metric-value">{fertilizer}</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="metric-tile green">
                <span class="metric-icon">📈</span>
                <div class="metric-label">{L['yield_label']}</div>
                <div class="metric-value">{expected_yield:,} {L['kg_hectare']}</div>
            </div>
        """, unsafe_allow_html=True)
    with mcol2:
        st.markdown(f"""
            <div class="metric-tile blue">
                <span class="metric-icon">💰</span>
                <div class="metric-label">{L['price_label']}</div>
                <div class="metric-value">₹{price}/kg</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="metric-tile">
                <span class="metric-icon">💵</span>
                <div class="metric-label">{L['profit_label']}</div>
                <div class="metric-value">₹{profit:,}</div>
            </div>
        """, unsafe_allow_html=True)

    st.balloons()

# ============================================================
# FOOTER
# ============================================================
st.markdown(f"""
    <div class="footer-note">{L['footer']}</div>
""", unsafe_allow_html=True)
