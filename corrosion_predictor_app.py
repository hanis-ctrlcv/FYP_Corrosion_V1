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
        return "🟢low"
    elif rate <= 1.0:
        return "🟡medium"
    else:
        return "🔴hign"

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

# Keep region selection persistent
if "selected_region" not in st.session_state:
    st.session_state.selected_region = "Peninsular"

selected_region = st.selectbox(
    "Select Region:",
    regions,
    index=regions.index(st.session_state.selected_region),
    key="region_selector"
)
st.session_state.selected_region = selected_region

# Assign 20 pipes across 3 regions (10 + 5 + 5)
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

# Default selection
if "selected_pipe_name" not in st.session_state:
    st.session_state.selected_pipe_name = None

# Display color-coded clickable buttons
for i, pipe in enumerate(region_pipes):
    pipe_df = PIPE_DATA[pipe].iloc[0]
    rate = pipe_df["Pred_Ensemble(mm/yr)"]
    color = get_color(rate)
    emoji = get_severity(rate)

    btn_label = f"{pipe}\n{emoji}"

    # Use Streamlit native button for state update
    with cols[i % 5]:
        if st.button(btn_label, key=f"pipe_{pipe}", help=f"Click to view {pipe} details"):
            st.session_state.selected_pipe_name = pipe
            st.session_state.selected_pipe = pipe_df.to_dict()

# --- PIPE DETAILS PANEL ---
if st.session_state.selected_pipe_name:
    sel = st.session_state.selected_pipe
    sel_name = st.session_state.selected_pipe_name
    rate = sel["Pred_Ensemble(mm/yr)"]
    color = get_color(rate)
    severity = get_severity(rate)

    # Remaining Life Calculation
    T_CURRENT, T_MIN, MAE, PITTING_FACTOR = 10.0, 5.0, 0.0915, 1.5
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
# SECTION 4 — CSV UPLOAD FOR BULK PREDICTION
# ============================================================
st.subheader("📥 Upload CSV for Bulk Prediction")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    user_df = pd.read_csv(uploaded_file)
    st.dataframe(user_df.head(), use_container_width=True)

    if st.button("🔮 Run Predictions on uploaded CSV"):
        try:
            expected = list(preprocessor.feature_names_in_)
            user_df_reindexed = user_df.reindex(columns=expected)
            X_new = preprocessor.transform(user_df_reindexed)

            p_dl = model_dl.predict(X_new).ravel()
            p_rf = rf.predict(X_new)
            p_xgb = xgb.predict(X_new)
            p_ens = (p_dl + p_rf + p_xgb) / 3

            result_df = pd.DataFrame({
                "Environment": user_df.get("Environment", None),
                "Material Family": user_df.get("Material Family", None),
                "Concentration_%": user_df.get("Concentration_%", None),
                "Temperature_C": user_df.get("Temperature_C", None),
                "Pred_DL(mm/yr)": p_dl,
                "Pred_RF(mm/yr)": p_rf,
                "Pred_XGB(mm/yr)": p_xgb,
                "Pred_Reinforced_DL(mm/yr)": p_ens,
                "Severity": [get_severity(v) for v in p_ens]
            })

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("predictions", exist_ok=True)
            out_path = f"predictions/prediction_{timestamp}.csv"
            result_df.to_csv(out_path, index=False)
            st.success(f"✅ Prediction Complete — Saved to {out_path}")
            st.dataframe(result_df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# ============================================================
# SECTION 5 — MODEL VISUALIZATION & ACCURACY
# ============================================================
st.subheader("📊 Model Prediction Comparison (training dataset)")
X = df.drop(columns=["Rate (mm/yr)"], errors="ignore")
y = df["Rate (mm/yr)"] if "Rate (mm/yr)" in df.columns else df.iloc[:, 0]  # fallback
X_prepared = preprocessor.transform(X)

y_dl = model_dl.predict(X_prepared).ravel()
y_rf = rf.predict(X_prepared)
y_xgb = xgb.predict(X_prepared)
y_ens = (y_dl + y_rf + y_xgb) / 3

df_viz = pd.DataFrame({
    "Actual": y,
    "Deep Learning": y_dl,
    "Random Forest": y_rf,
    "XGBoost": y_xgb,
    "Reinforced Deep Learning": y_ens
})

melted = df_viz.melt(id_vars="Actual", var_name="Model", value_name="Predicted")
fig = px.scatter(melted, x="Actual", y="Predicted", color="Model", title="Actual vs Predicted Corrosion Rate")
fig.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="black", dash="dash"))
st.plotly_chart(fig, use_container_width=True)

# Accuracy summary
r2_val = r2_score(y, y_ens)
mae_val = mean_absolute_error(y, y_ens)
rmse_val = np.sqrt(mean_squared_error(y, y_ens))
accuracy_pct = r2_val * 100

st.markdown(f"""
<div style='background-color:#E8F6EF;padding:15px;border-radius:10px;margin-top:10px;'>
    <h4 style='text-align:center;color:#1E8449;'>
        ✅ <b>Reinforced Deep Learning Accuracy: {accuracy_pct:.2f}%</b><br>
        (R² = {r2_val:.4f}, MAE = {mae_val:.4f}, RMSE = {rmse_val:.4f})
    </h4>
</div>
""", unsafe_allow_html=True)
# ============================================================
# CORRELATION ANALYSIS BY SEVERITY & PIPE
# ============================================================

st.subheader("Average Corrosion Rate by Material Family")
# Average Corrosion Rate by Material Family
avg_rates = df.groupby("Material Family")["Rate (mm/yr)"].mean().sort_values()
fig = px.bar(avg_rates, x=avg_rates.index, y=avg_rates.values,
             title="Average Corrosion Rate by Material Family", color=avg_rates.values,
             color_continuous_scale="RdYlGn_r")
st.plotly_chart(fig, use_container_width=True)

st.subheader(" Correlation Heatmap of Features")
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("Feature Interaction Overview (Pairplot)")
selected_cols = [c for c in ["Rate (mm/yr)", "Concentration_%", "Temperature_C", "Aggressiveness_Index"] if c in df.columns]
if len(selected_cols) >= 2:
    sns.pairplot(df[selected_cols], diag_kind="kde", corner=True)
    st.pyplot(plt)
else:
    st.info("Not enough columns available for pairplot.")




