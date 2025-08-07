import streamlit as st
import numpy as np
import joblib
import gradio as gr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")  # Make sure to save and load scaler

# Gradio interface function
def predict_class(Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
                  V11, V12, V13, V14, V15, V16, V17, V18, V19,
                  V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount):
    try:
        # Create the feature array
        features = np.array([[Time, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                              V10, V11, V12, V13, V14, V15, V16, V17, V18,
                              V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount]])
        features_scaled = scaler.transform(features)  # Use the saved scaler
        prediction = model.predict(features_scaled)
        return "Fraudulent Transaction" if prediction[0] == 1 else "Genuine Transaction"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
inputs = [gr.Number(label=f"V{i}") for i in range(1, 29)] + [gr.Number(label="Amount")]
output = gr.Textbox(label="Prediction")

# Create the interface
interface = gr.Interface(fn=predict_class, inputs=inputs, outputs=output, title="Fraud Detection Predictor")

# Display Gradio interface in Streamlit
st.title("Fraud Detection using Logistic Regression")
st.write("This app predicts whether a transaction is fraudulent or genuine based on the provided features.")

# Embed the Gradio interface inside Streamlit
interface.launch(inline=True)
