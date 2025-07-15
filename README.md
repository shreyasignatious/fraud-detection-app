💳 Credit Card Fraud Detection App

This is a Streamlit-based web application that allows users to upload a credit card transaction CSV and get predictions on which transactions are fraudulent.

🔍 Features:
- Upload a CSV file (with Time, Amount, V1-V28, Country)
- Preprocessing: scaling, column reordering
- Fraud prediction using a trained XGBoost model
- SHAP explainability plot for the first detected fraud
- Fraud-by-country bar chart with flag emojis
- User-friendly, fast, and deployable on Streamlit Cloud

🧠 Technologies:
- Python, Streamlit, XGBoost, SHAP, Matplotlib, Pandas, Scikit-learn

📦 File Structure:
fraud-detection/
├── app/
│ └── app.py
├── models/
│ └── model.pkl
├── data/
│ └── creditcard_with_country.csv
├── requirements.txt
└── README.md

 🚀 Deployment:
1. Upload to GitHub
2. Deploy on https://streamlit.io/cloud with app path: `app/app.py`
