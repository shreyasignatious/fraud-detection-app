import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------------------- App Configuration ----------------------
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
st.title("💳 Credit Card Fraud Detection App")
st.markdown("Upload your CSV file and detect potential fraudulent transactions using a trained ML model.")

# ---------------------- Load Model ----------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/model.pkl")
    except FileNotFoundError:
        st.error("❌ Model file not found. Please check if 'models/model.pkl' exists.")
        return None

model = load_model()
if not model:
    st.stop()

# ---------------------- Upload CSV ----------------------
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Check required columns
    if 'Amount' not in df.columns or 'Time' not in df.columns:
        st.error("❌ The uploaded file must contain 'Amount' and 'Time' columns.")
        st.stop()

    # ---------------------- Data Preprocessing ----------------------
    df['scaled_amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    scaled_amount = df.pop('scaled_amount')
    scaled_time = df.pop('scaled_time')
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)

    # Drop unused columns
    features = df.drop(['Class', 'Country'], axis=1, errors='ignore')
    features = features.select_dtypes(include=['number', 'bool'])

    # ---------------------- Prediction ----------------------
    predictions = model.predict(features)
    df['Prediction'] = predictions

    st.subheader("🔍 Prediction Results")
    st.dataframe(df[['Prediction']])
    st.success(f"✅ Fraud Cases Detected: {sum(df['Prediction'])}")

    # ---------------------- Fraud by Country ----------------------
    if 'Country' in df.columns:
        st.subheader("🌍 Fraud Cases by Country")
        flag_map = {
            "India": "🇮🇳", "USA": "🇺🇸", "UK": "🇬🇧", "Germany": "🇩🇪",
            "France": "🇫🇷", "Canada": "🇨🇦", "Australia": "🇦🇺"
        }
        fraud_df = df[df['Prediction'] == 1]
        fraud_by_country = fraud_df['Country'].value_counts()
        fraud_by_country.index = [f"{flag_map.get(c, '')} {c}" for c in fraud_by_country.index]
        st.bar_chart(fraud_by_country)
    else:
        st.info("ℹ️ 'Country' column not found for country-level chart.")

else:
    st.info("👆 Please upload a CSV file to start prediction.")
