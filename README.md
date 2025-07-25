# 💳 Credit Card Fraud Detection App

This is a machine learning-powered web application that detects fraudulent credit card transactions. Users can upload a CSV file, get real-time fraud predictions, and view SHAP-based explanations to understand why a transaction was flagged.

Built as part of the **Elevate Internship Program 2025**.

---

## 🚀 Live Demo

🔗 [Click here to try the app](https://fraud-detection-app-named4btnexxrmgb8sa5xt.streamlit.app/)

---

## 📌 Features

- Upload your own transaction CSV file
- Detects fraudulent transactions using XGBoost
- SHAP explanation for model interpretability
- Country-wise fraud detection chart with emoji flags
- Metrics summary for fraud vs legitimate
- Clean and responsive Streamlit interface

---

## 🛠️ Tools & Technologies

- Python 3.10
- Streamlit
- XGBoost
- Scikit-learn
- SHAP
- Pandas, NumPy
- Matplotlib

---

## 📂 Project Structure

fraud-detection-app/
├── app/                     ✅ contains `app.py`
├── models/                  ✅ contains `model.pkl`
├── utils/, notebooks/       ✅ optional support folders
├── creditcard_with_country_sample.csv  ✅ sample input
├── requirements.txt         ✅ for deployment
├── README.md                
└── report.pdf     ✅ internship report 
