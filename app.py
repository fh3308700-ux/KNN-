{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "c629e3d4-9dd4-4574-b3cd-e7293b4cc2d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import gradio as gr\n",
    "import nbformat\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "47889c23-028e-4c45-9ee4-de13c5b2abe4",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(r\"C:\\Users\\fh330\\Desktop\\python\\creditcard.csv\")\n",
    "if 'Class' not in df.columns:\n",
    "    raise ValueError(\"Dataset must contain a 'Class' column for fraud detection.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "0df2d008-f68d-4295-a7c7-78cad8d9820d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X = df.drop(['Class'], axis=1)\n",
    "y = df['Class']\n",
    "x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "955a6095-5a6c-4ad8-8310-5e13934e3d88",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "model = LogisticRegression(max_iter=1000)\n",
    "model.fit(x_train, y_train)\n",
    "predictions = model.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aace8f4d-c0d0-42a4-98f9-80f62aec6e4d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n",
    "print(\"Accuracy:\", accuracy_score(y_test, predictions))\n",
    "print(classification_report(y_test, predictions))\n",
    "print(\"Confusion Matrix:\\n\", confusion_matrix(y_test, predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b446bdbe-f75b-4b72-835e-e5cd3d8997b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib\n",
    "joblib.dump(model, \"fraud_model.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "65527c61-2df3-4fff-ae32-16e7b36718af",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = joblib.load(\"fraud_model.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "50e054ea-2080-4fd2-b289-c387d5561bd1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import gradio as gr "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "93107a7e-4866-488f-99bb-4a29c3f41aad",
   "metadata": {},
   "outputs": [],
   "source": [
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
    "inputs = [\n",
    "    gr.Number(label=\"Time\"),\n",
    "    gr.Number(label=\"V1\"),\n",
    "    gr.Number(label=\"V2\"),\n",
    "    gr.Number(label=\"V3\"),\n",
    "    gr.Number(label=\"V4\"),\n",
    "    gr.Number(label=\"V5\"),\n",
    "    gr.Number(label=\"V6\"),\n",
    "    gr.Number(label=\"V7\"),\n",
    "    gr.Number(label=\"V8\"),\n",
    "    gr.Number(label=\"V9\"),\n",
    "    gr.Number(label=\"V10\"),\n",
    "    gr.Number(label=\"V11\"),\n",
    "    gr.Number(label=\"V12\"),\n",
    "    gr.Number(label=\"V13\"),\n",
    "    gr.Number(label=\"V14\"),\n",
    "    gr.Number(label=\"V15\"),\n",
    "    gr.Number(label=\"V16\"),\n",
    "    gr.Number(label=\"V17\"),\n",
    "    gr.Number(label=\"V18\"),\n",
    "    gr.Number(label=\"V19\"),\n",
    "    gr.Number(label=\"V20\"),\n",
    "    gr.Number(label=\"V21\"),\n",
    "    gr.Number(label=\"V22\"),\n",
    "    gr.Number(label=\"V23\"),\n",
    "    gr.Number(label=\"V24\"),\n",
    "    gr.Number(label=\"V25\"),\n",
    "    gr.Number(label=\"V26\"),\n",
    "    gr.Number(label=\"V27\"),\n",
    "    gr.Number(label=\"V28\"),\n",
    "    gr.Number(label=\"Amount\")\n",
    "]\n",
    "\n",
    "output = gr.Textbox(label=\"Prediction\")\n",
    "gr.Interface(fn=predict_class, inputs=inputs, outputs=output, title=\"Fraud Detection Predictor\").launch()"
   ]
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
