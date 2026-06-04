import streamlit as st
import pickle
import pandas as pd

# Load model + features
model = pickle.load(open("model/car_model.pkl", "rb"))
feature_columns = pickle.load(open("model/features.pkl", "rb"))

st.title("🚗 Car Price Prediction App")

st.write("Enter details below:")

# Inputs
year = st.number_input("Year")
present_price = st.number_input("Present Price (in Lakhs)")
kms = st.number_input("Kilometers Driven")
owner = st.number_input("Number of Owners")

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

if st.button("Predict Price"):

    # Base input dictionary
    input_dict = {
        "Year": year,
        "Present_Price": present_price,
        "Kms_Driven": kms,
        "Owner": owner,

        "Fuel_Type_Diesel": 0,
        "Fuel_Type_Petrol": 0,
        "Seller_Type_Individual": 0,
        "Transmission_Manual": 0
    }

    # Encoding
    if fuel == "Diesel":
        input_dict["Fuel_Type_Diesel"] = 1
    elif fuel == "Petrol":
        input_dict["Fuel_Type_Petrol"] = 1

    if seller == "Individual":
        input_dict["Seller_Type_Individual"] = 1

    if transmission == "Manual":
        input_dict["Transmission_Manual"] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Match training columns EXACTLY
    input_df = input_df.reindex(feature_columns, axis=1, fill_value=0)

    # Prediction
    prediction = model.predict(input_df)

    st.success(f"🚘 Predicted Price: ₹ {prediction[0]:.2f} Lakhs")