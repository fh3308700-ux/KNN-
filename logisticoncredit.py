{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ac830d6f-67f3-4ecd-87d6-a55d8225e211",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.9991573329588147\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00     56864\n",
      "           1       0.83      0.64      0.72        98\n",
      "\n",
      "    accuracy                           1.00     56962\n",
      "   macro avg       0.91      0.82      0.86     56962\n",
      "weighted avg       1.00      1.00      1.00     56962\n",
      "\n",
      "Confusion Matrix:\n",
      " [[56851    13]\n",
      " [   35    63]]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\fh330\\anaconda3\\Lib\\site-packages\\gradio\\utils.py:1054: UserWarning: Expected 30 arguments for function <function predict_class at 0x000001D7E8672F20>, received 29.\n",
      "  warnings.warn(\n",
      "C:\\Users\\fh330\\anaconda3\\Lib\\site-packages\\gradio\\utils.py:1058: UserWarning: Expected at least 30 arguments for function <function predict_class at 0x000001D7E8672F20>, received 29.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "* Running on local URL:  http://127.0.0.1:7865\n",
      "* To create a public link, set `share=True` in `launch()`.\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div><iframe src=\"http://127.0.0.1:7865/\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": []
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "import gradio as gr\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n",
    "\n",
    "# Load data\n",
    "df = pd.read_csv(r\"C:\\Users\\fh330\\Desktop\\python\\creditcard.csv\")\n",
    "if 'Class' not in df.columns:\n",
    "    raise ValueError(\"Dataset must contain a 'Class' column for fraud detection.\")\n",
    "\n",
    "X = df.drop(['Class'], axis=1)\n",
    "y = df['Class']\n",
    "\n",
    "# Scale data\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "# Split data\n",
    "x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)\n",
    "\n",
    "# Train model\n",
    "model = LogisticRegression(max_iter=10000)\n",
    "model.fit(x_train, y_train)\n",
    "\n",
    "# Predict\n",
    "predictions = model.predict(x_test)\n",
    "\n",
    "# Evaluate\n",
    "print(\"Accuracy:\", accuracy_score(y_test, predictions))\n",
    "print(classification_report(y_test, predictions))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(y_test, predictions))\n",
    "\n",
    "# Save model\n",
    "joblib.dump(model, \"fraud_model.pkl\")\n",
    "\n",
    "# Load model\n",
    "model = joblib.load(\"fraud_model.pkl\")\n",
    "\n",
    "# Gradio interface\n",
    "def predict_class(Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,\n",
    "                  V11, V12, V13, V14, V15, V16, V17, V18, V19,\n",
    "                  V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount):\n",
    "    try:\n",
    "        features = np.array([[Time, V1, V2, V3, V4, V5, V6, V7, V8, V9,\n",
    "                              V10, V11, V12, V13, V14, V15, V16, V17, V18,\n",
    "                              V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount]])\n",
    "        prediction = model.predict(features)\n",
    "        return \"Fraudulent Transaction\" if prediction[0] == 1 else \"Genuine Transaction\"\n",
    "    except Exception as e:\n",
    "        return f\"Error: {str(e)}\"\n",
    "\n",
    "inputs = [gr.Number(label=f\"V{i}\") for i in range(1, 29)] + [gr.Number(label=\"Amount\")]\n",
    "output = gr.Textbox(label=\"Prediction\")\n",
    "gr.Interface(fn=predict_class, inputs=inputs, outputs=output, title=\"Fraud Detection Predictor\").launch()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f7c47f58-81b0-4dba-9ff0-0be405554601",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
