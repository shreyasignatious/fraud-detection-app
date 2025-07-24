import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap

# ---------------------- Custom Background ----------------------
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.transparenttextures.com/patterns/clean-gray-paper.png");
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# ---------------------- Sidebar ----------------------
st.sidebar.title("📌 About")
st.sidebar.info(
    "This app detects fraudulent credit card transactions using a trained ML model.\n\n"
    "💡 Upload your transaction CSV and get instant results.\n\n"
    "🔍 Built with XGBoost, Streamlit, SHAP, and sklearn."
)
st.sidebar.markdown("[📁 View on GitHub](https://github.com/shreyasignatious/fraud-detection-app)")

# ---------------------- Header ----------------------
st.markdown("<h1 style='text-align: center;'>💳 Credit Card Fraud Detection App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Upload a CSV file and detect potential fraud transactions instantly.</h4>", unsafe_allow_html=True)

# ---------------------- Load Model ----------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/model.pkl")
    except FileNotFoundError:
        st.error("❌ Model file not found in 'models/model.pkl'")
        return None

model = load_model()
if not model:
    st.stop()

# ---------------------- File Upload ----------------------
st.subheader("📂 Upload a CSV file")
uploaded_file = st.file_uploader("Upload a CSV to start prediction", type=["csv"])

# ---------------------- Download Sample CSV ----------------------
try:
    with open("creditcard_with_country_sample.csv", "rb") as file:
        st.download_button(label="📥 Download Sample CSV",
                           data=file,
                           file_name="sample_creditcard.csv",
                           mime="text/csv")
except FileNotFoundError:
    st.info("⚠️ Sample CSV not found in repository. Add 'creditcard_with_country_sample.csv' to enable this feature.")

# ---------------------- Prediction Workflow ----------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    if 'Amount' not in df.columns or 'Time' not in df.columns:
        st.error("❌ 'Amount' and 'Time' columns are required in the uploaded file.")
        st.stop()

    # Preprocessing
    df['scaled_amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    scaled_amount = df.pop('scaled_amount')
    scaled_time = df.pop('scaled_time')
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)

    # Features
    X = df.drop(['Class', 'Country'], axis=1, errors='ignore')
    X = X.select_dtypes(include=['number', 'bool'])

    # Prediction
    predictions = model.predict(X)
    df['Prediction'] = predictions

    # ---------------------- Metrics ----------------------
    st.subheader("📊 Summary")
    total = len(df)
    fraud = sum(df['Prediction'])
    st.metric("🧾 Total Transactions", total)
    st.metric("🚨 Fraudulent Transactions", fraud)
    st.metric("✅ Legitimate Transactions", total - fraud)

    # ---------------------- Results Table ----------------------
    st.subheader("🔍 Prediction Results")
    st.dataframe(df[['Prediction']])
    st.success(f"✅ Fraud Cases Detected: {fraud}")

    # ---------------------- SHAP Explanation ----------------------
    if fraud > 0:
        st.subheader("📈 SHAP Explanation for First Fraud Case")
        fraud_index = df[df['Prediction'] == 1].index[0]
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        fig, ax = plt.subplots()
        shap.plots._waterfall.waterfall_legacy(shap_values[fraud_index], max_display=10, show=False)
        st.pyplot(fig)
    else:
        st.info("✅ No fraud cases to explain with SHAP.")

    # ---------------------- Country-wise Chart ----------------------
    if 'Country' in df.columns:
        st.subheader("🌍 Fraud Cases by Country")
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
        st.info("ℹ️ 'Country' column not found to show fraud by country.")
else:
    st.info("📥 Please upload a CSV file above to begin fraud detection.")
