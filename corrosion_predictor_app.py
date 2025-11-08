# ============================================================
# Streamlit App: Corrosion Rate Prediction & Pipeline Simulation
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import datetime, os, base64, random

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="Pipeline Corrosion Status Dashboard", layout="wide")

# ============================================================
# HEADER
# ============================================================
logo_path = "utp logo.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <div style='text-align: center;'>
        <img src='data:image/png;base64,{logo_base64}' width='250'>
        <h2 style='margin-top: 10px; color:#004d80;'>Deep Learning Corrosion Prediction Dashboard</h2>
        <p style='color:#666;font-size:16px;'><b>Final Year Project by Muhammad Hanis Afifi Bin Azmi</b></p>
        <hr style='margin-top:10px;margin-bottom:30px;'>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_PATH = "final_corrosion_model.keras"
PREPROCESSOR_PATH = "preprocessor_corrosion.joblib"
RF_PATH = "rf_model.joblib"
XGB_PATH = "xgb_model.json"
DATA_PATH = "cleaned_corrosion_regression_data.csv"
SAMPLE_20_PATH = "random_20_samples.csv"

st.title("🛠️ Corrosion Monitoring Dashboard")
st.caption("Powered by Reinforced Deep Learning (DL + RF + XGB Ensemble) ✅")

# ============================================================
# LOAD MODELS & DATA SAFELY
# ============================================================
try:
    model_dl = tf.keras.models.load_model(MODEL_PATH, compile=False)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    rf = joblib.load(RF_PATH)
    xgb = XGBRegressor()
    xgb.load_model(XGB_PATH)
    df = pd.read_csv(DATA_PATH)
    st.success("✅ All models and dataset loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model or data: {e}")
    st.stop()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_severity(rate):
    if rate <= 0.1:
        return "🟢"
    elif rate <= 1.0:
        return "🟡"
    else:
        return "🔴"

def get_severity_label(rate):
    if rate <= 0.1:
        return "Low"
    elif rate <= 1.0:
        return "Medium"
    else:
        return "High"

def get_color(rate):
    if rate <= 0.1:
        return "#2ECC71"  # green
    elif rate <= 1.0:
        return "#F1C40F"  # yellow
    else:
        return "#E74C3C"  # red

# ============================================================
# PIPELINE FIXED DATA
# ============================================================
if os.path.exists(SAMPLE_20_PATH):
    sample_df = pd.read_csv(SAMPLE_20_PATH)
else:
    sample_df = df.sample(n=min(20, len(df)), random_state=42).reset_index(drop=True)

PIPE_DATA = {}
for i, row in sample_df.iterrows():
    PIPE_DATA[f"PIPE {i+1}"] = pd.DataFrame([{
        "Environment": row.get("Environment", "Unknown"),
        "Material Family": row.get("Material Family", "Unknown"),
        "Concentration_%": float(row.get("Concentration_%", 0)),
        "Temperature_C": float(row.get("Temperature_C", 0)),
        "Pred_DL(mm/yr)": float(row.get("Pred_DL(mm/yr)", 0)),
        "Pred_RF(mm/yr)": float(row.get("Pred_RF(mm/yr)", 0)),
        "Pred_XGB(mm/yr)": float(row.get("Pred_XGB(mm/yr)", 0)),
        "Pred_Ensemble(mm/yr)": float(row.get("Pred_Ensemble(mm/yr)", 0)),
        "Severity": get_severity_label(float(row.get("Pred_Ensemble(mm/yr)", 0)))
    }])

# ============================================================
# SECTION 1 — DATASET OVERVIEW
# ============================================================
st.subheader("📂 Trained Dataset Overview")
st.dataframe(df.head(), use_container_width=True)

# ============================================================
# SECTION 2 — REGIONAL PIPELINE OVERVIEW
# ============================================================
st.subheader("🌍 Regional Pipeline Overview")

regions = ["Peninsular", "Sabah", "Sarawak"]

# Keep selected region persistent between reruns
if "selected_region" not in st.session_state:
    st.session_state.selected_region = "Peninsular"

# If user switches manually, update local state only
selected_region = st.selectbox("Select Region:", regions, index=regions.index(st.session_state.selected_region))
st.session_state.selected_region = selected_region  # safe now (only 1 assignment per rerun)

# Assign pipes across regions
if "region_map" not in st.session_state:
    all_pipes = list(PIPE_DATA.keys())
    random.shuffle(all_pipes)
    st.session_state.region_map = {
        "Peninsular": all_pipes[:10],
        "Sabah": all_pipes[10:15],
        "Sarawak": all_pipes[15:20],
    }

region_pipes = st.session_state.region_map[selected_region]

st.markdown(f"### 🚧 {selected_region} Pipeline Corrosion Status")
cols = st.columns(5)

if "selected_pipe" not in st.session_state:
    st.session_state.selected_pipe = None
    st.session_state.selected_pipe_name = None

# Show pipe buttons
for i, pipe in enumerate(region_pipes):
    pipe_df = PIPE_DATA[pipe].iloc[0]
    rate = pipe_df["Pred_Ensemble(mm/yr)"]
    color = get_color(rate)
    emoji = get_severity(rate)
    label = get_severity_label(rate)
    btn_html = f"""
        <button onclick="window.location.search='pipe={pipe}'"
            style="
                background-color: {color};
                color: white;
                font-weight: bold;
                border-radius: 10px;
                border: none;
                width: 120px;
                height: 90px;
                font-size: 15px;
                margin: 6px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.3);
                cursor: pointer;
                transition: 0.2s all ease-in-out;">
            {pipe}<br>{emoji} {label}
        </button>
    """
    with cols[i % 5]:
        st.markdown(btn_html, unsafe_allow_html=True)

# Handle URL selection without refreshing region
query_params = st.experimental_get_query_params()
if "pipe" in query_params:
    selected_pipe = query_params["pipe"][0]
    if selected_pipe in PIPE_DATA:
        st.session_state.selected_pipe = PIPE_DATA[selected_pipe].iloc[0].to_dict()
        st.session_state.selected_pipe_name = selected_pipe

# --- PIPE DETAILS PANEL ---
if st.session_state.selected_pipe is not None:
    sel = st.session_state.selected_pipe
    sel_name = st.session_state.selected_pipe_name
    rate = sel["Pred_Ensemble(mm/yr)"]
    color = get_color(rate)
    severity = get_severity(rate)

    # Remaining life
    T_CURRENT, T_MIN, MAE, PITTING_FACTOR = 10.0, 5.0, 0.15, 1.5
    r_eff = (rate + MAE) * PITTING_FACTOR
    life = (T_CURRENT - T_MIN) / r_eff if r_eff > 0 else np.inf

    st.markdown(f"""
    <div style='background-color:#f7f9f9;border-radius:12px;padding:15px;
    box-shadow:0 4px 10px rgba(0,0,0,0.1);margin-top:20px;'>
        <h4 style='text-align:center;color:{color};'>
            📊 Selected Pipe Details — <b>{sel_name} ({selected_region})</b>
        </h4>
        <p style='font-size:16px;line-height:1.6;'>
         <b>Environment:</b> {sel['Environment']}<br>
         <b>Material:</b> {sel['Material Family']}<br>
         <b>Concentration:</b> {sel['Concentration_%']:.2f} %<br>
         <b>Temperature:</b> {sel['Temperature_C']:.2f} °C<br><br>
        🔹 <b>Pred_DL(mm/yr):</b> {sel['Pred_DL(mm/yr)']:.4f}<br>
        🔹 <b>Pred_RF(mm/yr):</b> {sel['Pred_RF(mm/yr)']:.4f}<br>
        🔹 <b>Pred_XGB(mm/yr):</b> {sel['Pred_XGB(mm/yr)']:.4f}<br>
        🔹 <b>Pred_Ensemble(mm/yr):</b> {rate:.4f}<br><br>
         <b>Severity:</b> <span style='color:{color};font-weight:bold;'>{severity}</span><br>
         <b>Estimated Remaining Life:</b> {life:.2f} years
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# REGION SUMMARY TABLE
# ============================================================
st.subheader("📋 Region Pipe Summary")
summary = []
for p in region_pipes:
    d = PIPE_DATA[p].iloc[0]
    summary.append({
        "Pipe": p,
        "Environment": d["Environment"],
        "Material": d["Material Family"],
        "Concentration_%": f"{d['Concentration_%']:.2f}",
        "Temperature_C": f"{d['Temperature_C']:.2f}",
        "Pred_Ensemble(mm/yr)": d["Pred_Ensemble(mm/yr)"],
        "Severity": d["Severity"]
    })
st.dataframe(pd.DataFrame(summary), use_container_width=True)

# ============================================================
# MODEL PERFORMANCE VISUALIZATION
# ============================================================
st.subheader("📊 Model Prediction Comparison")
X = df.drop(columns=["Rate (mm/yr)"], errors="ignore")
y = df["Rate (mm/yr)"] if "Rate (mm/yr)" in df.columns else df.iloc[:, 0]
X_prep = preprocessor.transform(X)

y_dl = model_dl.predict(X_prep).ravel()
y_rf = rf.predict(X_prep)
y_xgb = xgb.predict(X_prep)
y_ens = (y_dl + y_rf + y_xgb) / 3

viz = pd.DataFrame({"Actual": y, "Reinforced DL": y_ens})
fig = px.scatter(viz, x="Actual", y="Reinforced DL", title="Actual vs Predicted (Reinforced DL)")
fig.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="black", dash="dash"))
st.plotly_chart(fig, use_container_width=True)

r2, mae, rmse = r2_score(y, y_ens), mean_absolute_error(y, y_ens), np.sqrt(mean_squared_error(y, y_ens))
st.success(f"✅ R²={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")

# ============================================================
# CORRELATION HEATMAP
# ============================================================
st.subheader("📈 Correlation Heatmap")
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
