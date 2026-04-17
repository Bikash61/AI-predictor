# 🚀 AI Customer Churn Prediction System

An end-to-end Machine Learning web application that predicts customer churn risk using Artificial Neural Networks (ANN) and provides real-time insights through an interactive Streamlit dashboard.

---

## 📌 Project Overview

Customer churn is a critical problem in the banking and telecom industries. This project aims to identify customers who are likely to leave (churn) based on their profile and account details.

The system uses a trained Machine Learning model to analyze user inputs and predict:
- Churn Probability (%)
- Retention Probability (%)
- Risk Category (Low / Medium / High)

---

## 🎯 Objectives

- Build a predictive model using customer data
- Deploy the model as a real-time web application
- Provide an intuitive UI for non-technical users
- Visualize predictions using charts and metrics

---

## 🧠 Machine Learning Model

- Model Used: **Artificial Neural Network (MLPClassifier)**
- Preprocessing: **StandardScaler (feature scaling)**
- Training Data: Customer dataset (bank churn data)
- Output: Binary classification (Churn / Not Churn)

---

## ⚙️ Tech Stack

| Category        | Technology Used |
|----------------|----------------|
| Language       | Python         |
| ML Library     | Scikit-learn   |
| Framework      | Streamlit      |
| Data Handling  | Pandas, NumPy  |
| Deployment     | Streamlit Cloud |

---

## 🌐 Live Demo



## 📊 Features

✅ Real-time churn prediction  
✅ Probability-based output (Churn % & Retention %)  
✅ Risk classification (Low / Medium / High)  
✅ Interactive UI using Streamlit  
✅ Data visualization using charts  
✅ Clean and user-friendly interface  

---

## 🖥️ Application Workflow

1. User enters customer details (credit score, age, balance, etc.)
2. Input data is preprocessed using StandardScaler
3. Trained ANN model predicts churn probability
4. Results are displayed:
   - Churn Risk (%)
   - Retention Chance (%)
   - Risk Level
5. Visualization is shown using charts

---

## 📁 Project Structure
churn_project/
│
├── app.py # Streamlit application
├── model.pkl # Trained ML model
├── scaler.pkl # Data preprocessing scaler
├── requirements.txt # Dependencies
├── dataset.csv # Training dataset
└── README.md # Project documentation


---

## 🔧 Installation & Setup

### Step 1: Clone the repository
```bash
git clone https://github.com/your-username/churn-streamlit.git
cd churn-streamlit

Step 2: Install dependencies
pip install -r requirements.txt
Step 3: Run the application
streamlit run app.py

📈 Sample Output
Churn Risk: 10.35%
Retention Chance: 89.65%
Final Decision: LOW RISK

🚀 Future Improvements
Add more features (gender, geography, etc.)
Improve model accuracy using deep learning (TensorFlow)
Add user authentication system
Store prediction history using database
Deploy using Docker / AWS

📚 Learning Outcomes
Hands-on experience with Machine Learning pipeline
Model training and evaluation
Web app development using Streamlit
Cloud deployment of ML models
Real-world problem solving

👨‍💻 Author
Bikash Bashyal

GitHub: https://github.com/Mokarram7

⭐ Acknowledgement
This project was developed as part of academic learning to understand Machine Learning deployment and real-world applications.
