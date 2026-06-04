import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("model/car_model.pkl", "rb"))
features = pickle.load(open("model/features.pkl", "rb"))

st.title("🚗 Car Price Prediction")

# input
year = st.number_input("Year")
present_price = st.number_input("Present Price")
kms = st.number_input("KMs Driven")
owner = st.number_input("Owner")

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

if st.button("Predict"):

    # Create empty dataframe
    input_dict = {col: 0 for col in features}

    # fill numeric values
    if "Year" in input_dict: input_dict["Year"] = year
    if "Present_Price" in input_dict: input_dict["Present_Price"] = present_price
    if "Kms_Driven" in input_dict: input_dict["Kms_Driven"] = kms
    if "Owner" in input_dict: input_dict["Owner"] = owner

    # fill categorical
    if f"Fuel_Type_{fuel}" in input_dict:
        input_dict[f"Fuel_Type_{fuel}"] = 1

    if f"Seller_Type_{seller}" in input_dict:
        input_dict[f"Seller_Type_{seller}"] = 1

    if f"Transmission_{transmission}" in input_dict:
        input_dict[f"Transmission_{transmission}"] = 1

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Price: ₹ {prediction:.2f} Lakhs")