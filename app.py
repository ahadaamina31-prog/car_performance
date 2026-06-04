import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model/car_model.pkl", "rb"))

st.title("🚗 Car Price Prediction App")

st.write("Enter details below:")

year = st.number_input("Year")
present_price = st.number_input("Present Price (in Lakhs)")
kms = st.number_input("Kilometers Driven")
owner = st.number_input("Number of Owners")

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

if st.button("Predict Price"):

    # Manual encoding (same as training logic)
    fuel_diesel = 1 if fuel == "Diesel" else 0
    fuel_petrol = 1 if fuel == "Petrol" else 0

    seller_ind = 1 if seller == "Individual" else 0
    trans_manual = 1 if transmission == "Manual" else 0

    input_data = np.array([[
        year,
        present_price,
        kms,
        owner,
        fuel_diesel,
        fuel_petrol,
        seller_ind,
        trans_manual
    ]])

    prediction = model.predict(input_data)

    st.success(f"🚘 Predicted Price: ₹ {prediction[0]:.2f} Lakhs")