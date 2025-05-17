import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load model and scaler ---
MODEL_PATH = os.path.join('models', 'model.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --- Define features ---
FEATURES = [
    'price', '1h', '24h', '7d', '24h_volume', 'mkt_cap',
    'log_mkt_cap', 'log_24h_volume'
]

st.title("Crypto Liquidity Prediction App")

# Collect user input
user_input = {}
st.header("Enter Cryptocurrency Features:")
for feat in FEATURES:
    user_input[feat] = st.number_input(f"{feat.replace('_',' ').title()}:", value=0.0)

if st.button("Predict Liquidity"):
    try:
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        st.success(f"Predicted Liquidity: {round(float(prediction), 6)}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# --- Visualization Section ---
st.header("Visualizations")

# Upload dataset for visualization
uploaded_file = st.file_uploader("Upload a CSV file for visualization", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display dataset
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.warning("No numeric columns available for correlation heatmap.")
    else:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)

    # Feature distributions
    st.subheader("Feature Distributions")
    for column in df.select_dtypes(include=[np.number]).columns:
        st.write(f"Distribution of {column}")
        plt.figure()
        sns.histplot(df[column], kde=True, color='blue')
        st.pyplot(plt)

    