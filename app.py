import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="AI Fraud Detection", layout="wide")

st.title("💳 AI Fraud Detection Dashboard")
st.caption("Demo app to estimate fraud risk using transaction patterns")

# -------------------------------
# Load artifacts
# -------------------------------
if not os.path.exists("model.pkl"):
    st.error("❌ Model not found. Run train_model.py first.")
    st.stop()

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# -------------------------------
# City map (simulated)
# -------------------------------
city_coords = {
    "Delhi": (28.61, 77.20),
    "Mumbai": (19.07, 72.87),
    "Lucknow": (26.85, 80.95),
    "Bangalore": (12.97, 77.59),
    "Kolkata": (22.57, 88.36)
}

# -------------------------------
# Layout
# -------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("🧾 Transaction")
    amt = st.number_input("Amount (₹)", 0.0)
    hour = st.slider("Hour", 0, 23)
    city_pop = st.number_input("City Population", 0)

with right:
    st.subheader("📍 Locations")
    user_city = st.selectbox("User City", list(city_coords.keys()))
    merchant_city = st.selectbox("Merchant City", list(city_coords.keys()))

# -------------------------------
# Predict
# -------------------------------
if st.button("🚀 Predict"):
    lat, long = city_coords[user_city]
    merch_lat, merch_long = city_coords[merchant_city]

    distance = np.sqrt((lat - merch_lat)**2 + (long - merch_long)**2)
    is_night = 1 if hour < 6 else 0
    high_amt = 1 if amt > 500 else 0

    # Build input
    input_dict = {col: 0 for col in columns}
    for key in ["amt", "city_pop", "hour", "distance", "is_night", "high_amt"]:
        if key in input_dict:
            input_dict[key] = locals()[key]

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    prob = model.predict_proba(input_scaled)[0][1]

    # -------------------------------
    # Dashboard Output
    # -------------------------------
    st.markdown("---")

    colA, colB, colC = st.columns(3)

    colA.metric("Fraud Probability", f"{prob:.2%}")
    colB.metric("Distance", f"{distance:.2f}")
    colC.metric("Risk Level", "HIGH" if prob > 0.3 else "LOW")

    st.progress(int(prob * 100))

    if prob > 0.3:
        st.error("🚨 High Fraud Risk")
    elif prob > 0.15:
        st.warning("⚠️ Medium Risk")
    else:
        st.success("✅ Low Risk")

    # -------------------------------
    # Chart (simple visualization)
    # -------------------------------
    st.markdown("### 📊 Risk Breakdown")

    chart_df = pd.DataFrame({
        "Factor": ["Amount", "Distance", "Night"],
        "Value": [amt, distance, is_night]
    })

    st.bar_chart(chart_df.set_index("Factor"))