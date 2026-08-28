import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load Model, Scaler, Label Encoder
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# Fertilizer Recommendation Function
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

# Yield Data (kg per hectare)
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

# Market Price (Rs per kg)
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

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="centered"
)

# ---- HEADER ----
st.markdown("""
    <h1 style='text-align:center; color:green;'>
    🌾 Smart Crop Recommendation System
    </h1>
    <p style='text-align:center; color:gray;'>
    Pavai Arts and Science College for Women
    </p>
    <p style='text-align:center; color:gray;'>
    BSc AI & DS | Team: M.S.Monica & A.Monisha
    </p>
    <hr>
""", unsafe_allow_html=True)

# ---- INPUT SECTION ----
st.subheader("📥 Enter Soil & Weather Details")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", 0, 200, 90)
    P = st.number_input("Phosphorus (P)", 0, 200, 42)
    K = st.number_input("Potassium (K)", 0, 200, 43)
    temperature = st.number_input(
        "Temperature (°C)", 0.0, 50.0, 20.8)

with col2:
    humidity = st.number_input(
        "Humidity (%)", 0.0, 100.0, 82.0)
    ph = st.number_input(
        "pH Level", 0.0, 14.0, 6.5)
    rainfall = st.number_input(
        "Rainfall (mm)", 0.0, 500.0, 202.9)

st.markdown("<br>", unsafe_allow_html=True)

# ---- PREDICT BUTTON ----
if st.button("🌾 Recommend Crop",
             use_container_width=True):

    # Prediction
    sample = pd.DataFrame(
        [[N, P, K, temperature,
          humidity, ph, rainfall]],
        columns=['N', 'P', 'K', 'temperature',
                 'humidity', 'ph', 'rainfall'])

    sample_scaled = scaler.transform(sample)
    pred_encoded = model.predict(sample_scaled)
    crop = le.inverse_transform(pred_encoded)[0]

    # Fertilizer, Yield, Profit
    fertilizer = fertilizer_recommend(N, P, K)
    expected_yield = yield_data.get(crop, 1000)
    price = market_price.get(crop, 30)
    profit = expected_yield * price

    # ---- RESULTS ----
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📊 Recommendation Results")

    st.success(f"✅ Recommended Crop: **{crop.upper()}**")

    col3, col4 = st.columns(2)
    with col3:
        st.info(f"💊 Fertilizer:\n{fertilizer}")
        st.info(f"📈 Expected Yield:\n{expected_yield} kg/hectare")
    with col4:
        st.info(f"💰 Market Price:\n₹{price}/kg")
        st.success(f"💵 Expected Profit:\n₹{profit:,}")

    st.snow()

# ---- FOOTER ----
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align:center; color:gray;'>
    Team: M.S. Monica & A. Monisha |
    Guide: Ms. Anithakumari |
    BSc AI & DS — 5th Semester
    </p>
""", unsafe_allow_html=True)
