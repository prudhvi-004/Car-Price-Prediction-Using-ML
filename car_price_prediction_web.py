# app.py

import streamlit as st
import numpy as np
import pickle

# ✅ Load the trained model
with open('car_price_model.pkl.gz', 'rb') as model_file:
    model = pickle.load(model_file)

# ✅ Load the fitted scaler
with open('car_price_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Prediction App")
st.markdown("Enter the car's features below to estimate its selling price.")

# Input fields
year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
fuel_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3, 'Electric': 4}
fuel_type_encoded = fuel_mapping[fuel_type]

seller_type = st.radio("Seller Type", ['Dealer', 'Individual'])
seller_encoded = 1 if seller_type == 'Dealer' else 0

transmission = st.radio("Transmission", ['Manual', 'Automatic'])
trans_encoded = 1 if transmission == 'Manual' else 0

mileage = st.number_input("Mileage (km/l)", min_value=0.0, step=0.1)
engine = st.number_input("Engine (CC)", min_value=500.0, step=10.0)
max_power = st.number_input("Max Power (bhp)", min_value=20.0, step=1.0)
seats = st.slider("Number of Seats", 2, 10, 5)

# Predict button
if st.button("Predict Price"):
    input_data = np.array([[year, km_driven, fuel_type_encoded, seller_encoded,
                            trans_encoded, mileage, engine, max_power, seats]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    st.success(f"💰 Estimated Selling Price: ₹{prediction[0]:,.2f}")
