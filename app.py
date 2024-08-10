import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(layout="wide")

scaler = joblib.load("Scaler.pkl")

st.title("Restaurant Rating Prediction")

st.caption("This app helps you to predict a restaurants review class")

st.divider()

averagecost= st.number_input("Please enter the estimated average cost for two", min_value=50, max_value=999999, value=1000, step=200)

tablebooking = st.selectbox("Restaurant has table booking?", ["Yes","No"])

onlinedelivery = st.selectbox("Restaurant has online booking?", ["Yes","No"])

pricerange = st.selectbox("what is the price range (1 cheapest, 4 most expensive)", [1,2,3,4])

predictbutton = st.button("Predict the review!")

st.divider()

model = joblib.load("mlmodel.pkl")

bookingstatus = 1 if tablebooking == "Yes" else 0

deliverystatus = 1 if onlinedelivery == "Yes" else 0

values = [[averagecost, bookingstatus, deliverystatus, pricerange]]
X_values = np.array(values)
X = scaler.transform(X_values)

if predictbutton:
    st.snow()
    
    prediction = model.predict(X)
    
    st.write(prediction)

