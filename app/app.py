import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap

# ---------------------- Sidebar ----------------------
st.sidebar.title("📌 About")
st.sidebar.info(
    "This app detects fraudulent credit card transactions using a trained XGBoost model.\n\n"
    "Built with Streamlit, scikit-learn, SHAP, and matplotlib."
)
st.sidebar.markdown("[📁 GitHub Repo](https://github.com/shreyasignatious/fraud-detection-app)")

# ---------------------- Header ----------------------
st.title("💳 Credit Card Fraud Detection App")

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
uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])

# ---------------------- Sample CSV ----------------------
try:
    with open("creditcard_with_country_sample.csv", "rb") as file:
        st.download_button("📥 Download Sample CSV", data=file, file_name="sample_creditcard.csv", mime="text/csv")
except FileNotFoundError:
    st.info("⚠️ Sample CSV not found in repo.")

# ---------------------- Prediction ----------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    if 'Amount' not in df.columns or 'Time' not in df.columns:
        st.error("❌ Missing 'Amount' or 'Time' column.")
        st.stop()

    # Preprocessing
    df['scaled_amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    scaled_amount = df.pop('scaled_amount')
    scaled_time = df.pop('scaled_time')
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)

    X = df.drop(['Class', 'Country'], axis=1, errors='ignore')
    X = X.select_dtypes(include=['number', 'bool'])

    # Prediction
    predictions = model.predict(X)
    df['Prediction'] = predictions
    fraud = sum(df['Prediction'])

    if fraud > 0:
        st.warning(f"🚨 {fraud} Fraud Cases Detected")
    else:
        st.success("✅ No fraudulent transactions found.")

    # Metrics
    st.subheader("📊 Summary")
    total = len(df)
    st.metric("🧾 Total Transactions", total)
    st.metric("🚨 Fraudulent", fraud)
    st.metric("✅ Legitimate", total - fraud)

    # Results Table
    st.subheader("🔍 Prediction Results")
    st.dataframe(df[['Prediction']])

    # SHAP
    if fraud > 0:
        st.subheader("📈 SHAP Explanation (First Fraud)")
        fraud_index = df[df['Prediction'] == 1].index[0]
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        fig, ax = plt.subplots()
        shap.plots._waterfall.waterfall_legacy(shap_values[fraud_index], max_display=10, show=False)
        st.pyplot(fig)

    # Country-wise Chart
    if 'Country' in df.columns:
        st.subheader("🌍 Fraud by Country")
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
        st.info("ℹ️ 'Country' column not found.")
else:
    st.info("📥 Please upload a CSV file above to begin fraud detection.")
