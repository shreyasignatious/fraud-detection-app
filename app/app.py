import shap
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load('../models/model.pkl')

# App title
st.title("💳 Credit Card Fraud Detection App")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a CSV file for prediction", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data Preview:")
    st.dataframe(df.head())

    if 'Amount' in df.columns and 'Time' in df.columns:
        # Scale 'Amount' and 'Time'
        df['scaled_amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))
        df.drop(['Time', 'Amount'], axis=1, inplace=True)

        # Reorder scaled columns
        scaled_amount = df['scaled_amount']
        scaled_time = df['scaled_time']
        df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
        df.insert(0, 'scaled_amount', scaled_amount)
        df.insert(1, 'scaled_time', scaled_time)

        # ✅ Drop 'Country' before prediction to avoid feature mismatch
        X = df.drop(['Class', 'Country'], axis=1, errors='ignore')
        X = X.select_dtypes(include=['number', 'bool'])  # Keep only numeric features

        # 🔮 Make predictions
        predictions = model.predict(X)
        df['Prediction'] = predictions

        # 📊 SHAP explanation (only for first fraud)
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        fraud_index = df[df['Prediction'] == 1].index
        if not fraud_index.empty:
            st.subheader("📊 SHAP Explanation (First Fraud Prediction)")
            st.write("This explains why the model predicted fraud.")
            shap_plot = shap.plots._waterfall.waterfall_legacy(
                shap_values[fraud_index[0]], max_display=10, show=False
            )
            st.pyplot(plt.gcf())
        else:
            st.info("✅ No fraud detected to explain.")

        # 📋 Prediction Results
        st.subheader("🔍 Prediction Results")
        st.dataframe(df[['Prediction']])
        st.success(f"✅ Fraud Cases Detected: {sum(df['Prediction'])}")

        # 🌍 Fraud by Country Chart (with flags)
        if 'Country' in df.columns:
            st.subheader("🌍 Fraud Cases by Country (with Flags)")
            flag_map = {
                "India": "🇮🇳", "USA": "🇺🇸", "UK": "🇬🇧",
                "Germany": "🇩🇪", "France": "🇫🇷",
                "Canada": "🇨🇦", "Australia": "🇦🇺"
            }
            fraud_df = df[df['Prediction'] == 1]
            fraud_by_country = fraud_df['Country'].value_counts()
            fraud_by_country.index = [f"{flag_map.get(c, '')} {c}" for c in fraud_by_country.index]
            st.bar_chart(fraud_by_country)
        else:
            st.info("ℹ️ 'Country' column not found for country-level chart.")
    else:
        st.error("❌ Required columns 'Amount' and 'Time' not found in uploaded file.")
