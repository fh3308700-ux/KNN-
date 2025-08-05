
import pandas as pd
import numpy as np
import joblib
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load data
df = pd.read_csv("C:\Users\fh330\Desktop\python\creditcard.csv")
if 'Class' not in df.columns:
    raise ValueError("Dataset must contain a 'Class' column for fraud detection.")

X = df.drop(['Class'], axis=1)
y = df['Class']

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = LogisticRegression(max_iter=10000)
model.fit(x_train, y_train)

# Predict
predictions = model.predict(x_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print("Confusion Matrix:
", confusion_matrix(y_test, predictions))

# Save model
joblib.dump(model, "fraud_model.pkl")

# Load model (in case you need to reload in Streamlit)
model = joblib.load("fraud_model.pkl")

# Gradio interface function
def predict_class(Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
                  V11, V12, V13, V14, V15, V16, V17, V18, V19,
                  V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount):
    try:
        # Create the feature array
        features = np.array([[Time, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                              V10, V11, V12, V13, V14, V15, V16, V17, V18,
                              V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount]])
        prediction = model.predict(features)
        return "Fraudulent Transaction" if prediction[0] == 1 else "Genuine Transaction"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
inputs = [gr.Number(label=f"V{i}") for i in range(1, 29)] + [gr.Number(label="Amount")]
output = gr.Textbox(label="Prediction")

gr.Interface(fn=predict_class, inputs=inputs, outputs=output, title="Fraud Detection Predictor").launch()
