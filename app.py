import streamlit as st
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

# Auto-generate model if pkl files are missing (e.g. on first cloud run)
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier

    csv_path = os.path.join(BASE_DIR, 'Artificial_Neural_Network_Case_Study_data.csv')
    df = pd.read_csv(csv_path)
    X = df[['CreditScore', 'Age', 'Balance', 'Tenure', 'EstimatedSalary']]
    y = df['Exited']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    pickle.dump(model, open(model_path, 'wb'))
    pickle.dump(scaler, open(scaler_path, 'wb'))

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

# Title
st.set_page_config(page_title="AI Churn Predictor", layout="centered")
st.title("🚀 AI Customer Churn Analyzer")

st.markdown("Fill the details below to predict customer churn risk")

# Inputs
creditscore = st.number_input("Credit Score", min_value=300, max_value=900)
age = st.number_input("Age", min_value=18, max_value=100)
balance = st.number_input("Balance")
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10)
salary = st.number_input("Estimated Salary")

# Button
if st.button("Analyze Risk"):

    # Prepare data
    data = np.array([[creditscore, age, balance, tenure, salary]])
    data = scaler.transform(data)

    # Prediction
    prediction = model.predict(data)
    prob = model.predict_proba(data)

    probability = round(prob[0][1] * 100, 2)
    retention = round(100 - probability, 2)

    # Output
    st.subheader("📊 Result")

    col1, col2 = st.columns(2)

    col1.metric("⚠️ Churn Risk", f"{probability}%")
    col2.metric("✅ Retention", f"{retention}%")

    # Decision
    if probability < 30:
        st.success("✅ LOW RISK")
    elif probability < 70:
        st.warning("⚠️ MEDIUM RISK")
    else:
        st.error("❌ HIGH RISK")

    # Progress bar
    st.progress(int(probability))

    # Chart
    st.subheader("📊 Risk Distribution")
    st.bar_chart({
        "Churn": [probability],
        "Retention": [retention]
    })