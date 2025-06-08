import streamlit as st
import pandas as pd
import mlflow.pyfunc
import pickle

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Set page config
st.set_page_config(
    page_title="🔋 Energy Price Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS Styling
st.markdown("""
    <style>
        html, body, .main {
            background-color: #f4f6f9;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        h1, h2, h3 {
            font-family: 'Segoe UI', sans-serif;
            color: #2c3e50;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .metric-container {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            text-align: center;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown("## ⚡ Energy Price Prediction Dashboard")
st.markdown("Use a trained ML model to estimate **electricity prices** based on selected parameters.")

# Load model
model_uri = "runs:/0da4b5c3d4984946a5e9950d9aca28f4/model"
model = mlflow.pyfunc.load_model(model_uri)

# Load vectorizer
with open("data/processed/dv.pkl", "rb") as f_in:
    dv = pickle.load(f_in)

# Sidebar inputs
st.sidebar.markdown("## 🧾 Input Parameters")
state = st.sidebar.selectbox("🌍 State", [
    "CA", "TX", "NY", "FL", "PA", "IL", "WI", "GA", "OH", "MI"
])
sector = st.sidebar.selectbox("🏢 Sector", [
    "Residential", "Commercial", "Industrial", "Transportation"
])

capability = st.sidebar.number_input("⚙️ Installed Capability (MW)", min_value=0.0, value=500.0, step=50.0)

# Prediction logic
if st.sidebar.button("🚀 Predict Price"):
    input_dict = {
        "stateid": state,
        "sectorName": sector,
        "capability": capability
    }
    input_transformed = dv.transform([input_dict])
    predicted_price = model.predict(input_transformed)[0]

    st.markdown("---")
    st.markdown("### 💡 Prediction Result")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div style="background-color:#ffffff; padding:20px; border-radius:12px; box-shadow:0 4px 10px rgba(0,0,0,0.05); text-align:center">
                <div style="font-size:40px;">📊</div>
                <div style="font-weight:bold; font-size:18px; color:#555;">Predicted Price</div>
                <div style="font-size:26px; margin-top:10px; color:#2c3e50;"><strong>${predicted_price:.2f}</strong></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with col2:
            st.markdown(
                f"""
                <div style="background-color:#ffffff; padding:20px; border-radius:12px; box-shadow:0 4px 10px rgba(0,0,0,0.05); text-align:center">
                    <div style="font-size:40px;">🔍</div>
                    <div style="font-weight:bold; font-size:18px; color:#555;">Sector</div>
                    <div style="font-size:22px; margin-top:10px; color:#2c3e50;"><strong>{sector}</strong></div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div style="background-color:#ffffff; padding:20px; border-radius:12px; box-shadow:0 4px 10px rgba(0,0,0,0.05); text-align:center">
                    <div style="font-size:40px;">🌍</div>
                    <div style="font-weight:bold; font-size:18px; color:#555;">State</div>
                    <div style="font-size:22px; margin-top:10px; color:#2c3e50;"><strong>{state}</strong></div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col4:
            st.markdown(
                f"""
                <div style="background-color:#ffffff; padding:20px; border-radius:12px; box-shadow:0 4px 10px rgba(0,0,0,0.05); text-align:center">
                    <div style="font-size:40px;">⚙️</div>
                    <div style="font-weight:bold; font-size:18px; color:#555;">Capability</div>
                    <div style="font-size:22px; margin-top:10px; color:#2c3e50;"><strong>{capability} MW</strong></div>
                </div>
                """,
                unsafe_allow_html=True
            )


    # st.success("✅ Model loaded successfully from MLflow!")
else:
    st.info("Fill in the sidebar and hit **Predict Price** to see results.")
