import streamlit as st
import pickle
import numpy as np

# =========================
# 1. LOAD MODEL
# =========================
model = pickle.load(open("model/car_model.pkl", "rb"))

st.title("🚗 Car Price Prediction Dashboard")
st.write("Enter car details to predict selling price")

# =========================
# 2. USER INPUT
# =========================

year = st.number_input("Year of Purchase", 1990, 2026, step=1)
present_price = st.number_input("Present Price (in Lakhs)", 0.0, 50.0)
kms_driven = st.number_input("Kilometers Driven", 0, 500000, step=1000)

owner = st.selectbox("Number of Owners", [0, 1, 2, 3])

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

# =========================
# 3. CONVERT INPUT TO MODEL FORMAT
# =========================

# Manual encoding (must match training dummies)
fuel_diesel = 1 if fuel_type == "Diesel" else 0
fuel_petrol = 1 if fuel_type == "Petrol" else 0

seller_individual = 1 if seller_type == "Individual" else 0

trans_manual = 1 if transmission == "Manual" else 0

# Feature array (order must match training dataset)
input_data = np.array([[
    year,
    present_price,
    kms_driven,
    owner,
    fuel_diesel,
    fuel_petrol,
    seller_individual,
    trans_manual
]])

# =========================
# 4. PREDICTION BUTTON
# =========================

if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"🚘 Estimated Selling Price: ₹ {prediction[0]:.2f} Lakhs")