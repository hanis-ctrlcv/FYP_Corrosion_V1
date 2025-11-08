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
# HEADER WITH UTP LOGO
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
# MODEL CONFIGURATION (edit paths if needed)
# ============================================================
MODEL_PATH = "final_corrosion_model.keras"
PREPROCESSOR_PATH = "preprocessor_corrosion.joblib"
RF_PATH = "rf_model.joblib"
XGB_PATH = "xgb_model.json"
DATA_PATH = "cleaned_corrosion_regression_data.csv"
SAMPLE_20_PATH = "random_20_samples.csv"

st.title("Corrosion Monitoring Dashboard")
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
# Helper: severity and color mapping
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
# PIPELINE FIXED DATA (Now 20 Pipes from CSV)
# ============================================================
PIPE_DATA = {}

if os.path.exists(SAMPLE_20_PATH):
    sample_df = pd.read_csv(SAMPLE_20_PATH)
else:
    sample_df = df.sample(n=min(20, len(df)), random_state=42).reset_index(drop=True)

col_map = {c.lower(): c for c in sample_df.columns}

def col(name_options):
    for opt in name_options:
        key = opt.lower()
        if key in col_map:
            return col_map[key]
    return None

ENV_COL = col(["Environment"])
MAT_COL = col(["Material Family", "Material_Family"])
CONC_COL = col(["Concentration_%", "Concentration"])
TEMP_COL = col(["Temperature_C", "Temperature"])
DL_COL = col(["Pred_DL(mm/yr)", "Pred_DL"])
RF_COL = col(["Pred_RF(mm/yr)", "Pred_RF"])
XGB_COL = col(["Pred_XGB(mm/yr)", "Pred_XGB"])
ENS_COL = col(["Pred_Ensemble(mm/yr)", "Pred_Ensemble"])

if ENS_COL is None and DL_COL and RF_COL and XGB_COL:
    sample_df["Pred_Ensemble(mm/yr)"] = (
        sample_df[DL_COL].astype(float)
        + sample_df[RF_COL].astype(float)
        + sample_df[XGB_COL].astype(float)
    ) / 3
    ENS_COL = "Pred_Ensemble(mm/yr)"

for i, row in sample_df.iterrows():
    PIPE_DATA[f"PIPE {i+1}"] = pd.DataFrame([{
        "Environment": row[ENV_COL],
        "Material Family": row[MAT_COL],
        "Concentration_%": float(row[CONC_COL]),
        "Temperature_C": float(row[TEMP_COL]),
        "Pred_DL(mm/yr)": float(row[DL_COL]),
        "Pred_RF(mm/yr)": float(row[RF_COL]),
        "Pred_XGB(mm/yr)": float(row[XGB_COL]),
        "Pred_Ensemble(mm/yr)": float(row[ENS_COL]),
        "Severity": get_severity_label(row[ENS_COL])
    }])

# ============================================================
# SECTION 1 — DATASET OVERVIEW
# ============================================================
st.subheader("📂 Trained Dataset Overview")
st.dataframe(df.head(), use_container_width=True)

# ============================================================
# SECTION 2 — REGIONAL PIPELINE CORROSION STATUS
# ============================================================
st.subheader("🌍 Regional Pipeline Overview")

regions = ["Peninsular", "Sabah", "Sarawak"]

# region state persistence
if "selected_region" not in st.session_state:
    st.session_state.selected_region = regions[0]

query_params = st.experimental_get_query_params()
if "region" in query_params:
    reg = query_params["region"][0]
    if reg in regions:
        st.session_state.selected_region = reg

selected_region = st.selectbox(
    "Select Region:",
    regions,
    index=regions.index(st.session_state.selected_region),
    key="selected_region"
)


# Create new region_map
if "region_map" not in st.session_state:
    all_pipes = list(PIPE_DATA.keys())
    random.shuffle(all_pipes)
    st.session_state.region_map = {
        "Peninsular": all_pipes[:10],
        "Sabah": all_pipes[10:15],
        "Sarawak": all_pipes[15:20]
    }

region_pipes = st.session_state.region_map[selected_region]
st.markdown(f"### 🚧 {selected_region} Pipeline Corrosion Status")
cols = st.columns(5)

if "selected_pipe" not in st.session_state:
    st.session_state.selected_pipe = None

# --- Handle selected pipe ---
if "selected_pipe" in query_params:
    selected = query_params["selected_pipe"][0]
    if selected in PIPE_DATA:
        st.session_state.selected_pipe = PIPE_DATA[selected].iloc[0].to_dict()
        st.session_state.selected_pipe_name = selected

# if region param also exists
if "region" in query_params:
    reg = query_params["region"][0]
    if reg in regions:
        st.session_state.selected_region = reg
        selected_region = reg

# --- Show clickable boxes ---
for i, pipe in enumerate(region_pipes):
    pipe_df = PIPE_DATA[pipe].iloc[0]
    rate = pipe_df["Pred_Ensemble(mm/yr)"]
    color = get_color(rate)
    emoji = get_severity(rate)
    label = get_severity_label(rate)

    safe_region = selected_region
    highlight = "4px solid black" if st.session_state.get("selected_pipe_name") == pipe else "none"

    button_html = f"""
        <a href="?selected_pipe={pipe}&region={safe_region}" target="_self">
            <button type="button" style="
                background-color: {color};
                color: white;
                font-weight: bold;
                border-radius: 10px;
                border: {highlight};
                padding: 10px;
                width: 120px;
                height: 90px;
                margin: 5px;
                font-size: 15px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.3);
                cursor: pointer;
                transition: 0.2s all ease-in-out;
            ">
                {pipe}<br>{emoji} {label}
            </button>
        </a>
    """
    with cols[i % 5]:
        st.markdown(button_html, unsafe_allow_html=True)

# --- PIPE DETAILS PANEL ---
if st.session_state.selected_pipe is not None:
    sel = st.session_state.selected_pipe
    sel_name = st.session_state.get("selected_pipe_name", "PIPE")
    rate = sel.get("Pred_Ensemble(mm/yr)", np.nan)
    color = get_color(rate)
    severity = get_severity(rate)

    T_CURRENT = 10.0
    T_MIN = 5.0
    MAE = 0.1187
    PITTING_FACTOR = 1.5

    r_eff = (max(rate, 0.0) + MAE) * PITTING_FACTOR
    life = (T_CURRENT - T_MIN) / r_eff if r_eff > 0 else np.inf

    st.markdown(f"""
    <div style='background-color:#f7f9f9;border-radius:12px;padding:15px;
    box-shadow:0 4px 10px rgba(0,0,0,0.1);margin-top:20px;'>
        <h4 style='text-align:center;color:{color};'>
            📊 Selected Pipe Details — <b>{sel_name} ({selected_region})</b>
        </h4>
        <p style='font-size:16px;line-height:1.6;'>
         <b>Environment:</b> {sel.get('Environment', '—')}<br>
         <b>Material:</b> {sel.get('Material Family', '—')}<br>
         <b>Concentration:</b> {sel.get('Concentration_%', 0):.2f} %<br>
         <b>Temperature:</b> {sel.get('Temperature_C', 0):.2f} °C<br><br>
        🔹 <b>Pred_DL(mm/yr):</b> {sel.get('Pred_DL(mm/yr)', np.nan):.4f}<br>
        🔹 <b>Pred_RF(mm/yr):</b> {sel.get('Pred_RF(mm/yr)', np.nan):.4f}<br>
        🔹 <b>Pred_XGB(mm/yr):</b> {sel.get('Pred_XGB(mm/yr)', np.nan):.4f}<br>
        🔹 <b>Pred_Ensemble(mm/yr):</b> {rate:.4f}<br><br>
         <b>Severity:</b> <span style='color:{color};font-weight:bold;'>{severity}</span><br>
         <b>Estimated Remaining Life:</b> {life:.2f} years
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# REMAINING SECTIONS (unchanged)
# ============================================================
# Keep your summary table, CSV upload, visualization, accuracy, etc.

