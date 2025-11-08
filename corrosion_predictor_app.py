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

# try importing stylable_container (optional). If not available we degrade gracefully.
try:
    from streamlit_extras.stylable_container import stylable_container
    STYLABLE_AVAILABLE = True
except Exception:
    STYLABLE_AVAILABLE = False

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
    """Return severity string (and emoji) based on corrosion rate."""
    if rate <= 0.1:
        return "🟢"
    elif rate <= 1.0:
        return "🟡"
    else:
        return "🔴"

def get_severity_label(rate):
    """Return simple label Low/Medium/High (no emoji)"""
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

# read the sample 20 rows CSV (must exist). If not, try to sample from main df.
if os.path.exists(SAMPLE_20_PATH):
    sample_df = pd.read_csv(SAMPLE_20_PATH)
else:
    # fallback: sample 20 rows from main df (if it's large enough)
    sample_df = df.sample(n=min(20, len(df)), random_state=42).reset_index(drop=True)

# Ensure column names exist; allow common alternatives
col_map = {c.lower(): c for c in sample_df.columns}

def col(name_options):
    """Find first matching column name from sample_df for a list of candidates."""
    for opt in name_options:
        key = opt.lower()
        if key in col_map:
            return col_map[key]
    return None

# candidate column names
ENV_COL = col(["Environment"])
MAT_COL = col(["Material Family", "Material_Family", "MaterialFamily"])
CONC_COL = col(["Concentration_%", "Concentration", "Concentration_%"])
TEMP_COL = col(["Temperature_C", "Temperature"])
DL_COL = col(["Pred_DL(mm/yr)", "Pred_DL", "Pred_DL(mm/yr)"])
RF_COL = col(["Pred_RF(mm/yr)", "Pred_RF", "Pred_RF(mm/yr)"])
XGB_COL = col(["Pred_XGB(mm/yr)", "Pred_XGB", "Pred_XGB(mm/yr)"])
ENS_COL = col(["Pred_Ensemble(mm/yr)", "Pred_Ensemble", "Pred_Ensemble(mm/yr)", "Pred_Ensemble(mm/yr)"])

# if ensemble column missing, try compute as average
if ENS_COL is None and DL_COL and RF_COL and XGB_COL:
    sample_df["Pred_Ensemble(mm/yr)"] = (
        sample_df[DL_COL].astype(float) +
        sample_df[RF_COL].astype(float) +
        sample_df[XGB_COL].astype(float)
    ) / 3
    ENS_COL = "Pred_Ensemble(mm/yr)"

# Build PIPE_DATA keys from sample rows (PIPE 1 ... PIPE N)
for i, row in sample_df.iterrows():
    env = row[ENV_COL] if ENV_COL else "Unknown"
    mat = row[MAT_COL] if MAT_COL else "Unknown"
    conc = float(row[CONC_COL]) if CONC_COL else np.nan
    temp = float(row[TEMP_COL]) if TEMP_COL else np.nan
    dl = float(row[DL_COL]) if DL_COL else np.nan
    rf_p = float(row[RF_COL]) if RF_COL else np.nan
    xgb_p = float(row[XGB_COL]) if XGB_COL else np.nan
    ens = float(row[ENS_COL]) if ENS_COL else np.nan

    PIPE_DATA[f"PIPE {i+1}"] = pd.DataFrame([{
        "Environment": env,
        "Material Family": mat,
        "Concentration_%": conc,
        "Temperature_C": temp,
        "Pred_DL(mm/yr)": dl,
        "Pred_RF(mm/yr)": rf_p,
        "Pred_XGB(mm/yr)": xgb_p,
        "Pred_Ensemble(mm/yr)": ens,
        "Severity": get_severity_label(ens)
    }])

# ============================================================
# SECTION 1 — DATASET OVERVIEW
# ============================================================
st.subheader(" Trained Dataset Overview")
st.dataframe(df.head(), use_container_width=True)

# ============================================================
# SECTION 2 — REGIONAL PIPELINE CORROSION STATUS
# ============================================================
st.subheader("🌍 Regional Pipeline Overview")

# Define regions (3 only)
regions = ["Peninsular", "Sabah", "Sarawak"]

# Region selector
selected_region = st.selectbox("Select Region:", regions)

# Create new region_map only if not already in session
if "region_map" not in st.session_state:
    all_pipes = list(PIPE_DATA.keys())
    random.shuffle(all_pipes)

    # Weighted distribution (10 + 5 + 5 = 20 pipes)
    peninsular_pipes = all_pipes[:10]
    sabah_pipes = all_pipes[10:15]
    sarawak_pipes = all_pipes[15:20]

    region_map = {
        "Peninsular": peninsular_pipes,
        "Sabah": sabah_pipes,
        "Sarawak": sarawak_pipes
    }

    st.session_state.region_map = region_map

region_pipes = st.session_state.region_map[selected_region]

st.markdown(f"### 🚧 {selected_region} Pipeline Corrosion Status")
cols = st.columns(5)

# create session_state.selected_pipe default
if "selected_pipe" not in st.session_state:
    st.session_state.selected_pipe = None

# Handle selection via query parameter for interactivity
query_params = st.experimental_get_query_params()
if "selected_pipe" in query_params:
    selected = query_params["selected_pipe"][0]  # get first value
    if selected in PIPE_DATA:
        st.session_state.selected_pipe = PIPE_DATA[selected].iloc[0].to_dict()
        st.session_state.selected_pipe_name = selected
# Show clickable colored boxes
for i, pipe in enumerate(region_pipes):
    pipe_df = PIPE_DATA[pipe].iloc[0]
    rate = pipe_df["Pred_Ensemble(mm/yr)"]
    color = get_color(rate)
    emoji = get_severity(rate)
    label = get_severity_label(rate)
    
    button_html = f"""
        <a href="?selected_pipe={pipe}" target="_self">
            <button type="button" style="
                background-color: {color};
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 10px;
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

    # Remaining Life Calculation (simple deterministic)
    T_CURRENT = st.sidebar.number_input("Current thickness (mm)", value=10.0, step=0.5)
    T_MIN = st.sidebar.number_input("Minimum allowable thickness (mm)", value=5.0, step=0.5)
    MAE = 0.15  # you may set based on model evaluation
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
# SECTION 3 — PIPE INFO DISPLAY (summarize all pipes in selected region)
# ============================================================
st.subheader("📋 Region Pipe Summary")
summary_rows = []
for p in region_pipes:
    r = PIPE_DATA[p].iloc[0]
    summary_rows.append({
        "Pipe": p,
        "Environment": r.get("Environment", ""),
        "Material Family": r.get("Material Family", ""),
        "Conc_%": f"{r.get('Concentration_%', 0):.2f}",
        "Temp_C": f"{r.get('Temperature_C', 0):.2f}",
        "Pred_Ensemble(mm/yr)": r.get("Pred_Ensemble(mm/yr)", np.nan),
        "Severity": r.get("Severity", "")
    })
summary_df = pd.DataFrame(summary_rows)
# color severity column
def style_sev(val):
    if str(val).lower().startswith("low"):
        return "color: green; font-weight: bold"
    if str(val).lower().startswith("medium"):
        return "color: orange; font-weight: bold"
    if str(val).lower().startswith("high"):
        return "color: red; font-weight: bold"
    return ""

st.dataframe(summary_df.style.applymap(style_sev, subset=["Severity"]), use_container_width=True)

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
# CORRELATION AND PAIRPLOT
# ============================================================
st.subheader("📈 Correlation Heatmap of Features")
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("🔍 Feature Interaction Overview (Pairplot)")
selected_cols = [c for c in ["Rate (mm/yr)", "Concentration_%", "Temperature_C", "Aggressiveness_Index"] if c in df.columns]
if len(selected_cols) >= 2:
    sns.pairplot(df[selected_cols], diag_kind="kde", corner=True)
    st.pyplot(plt)
else:
    st.info("Not enough columns available for pairplot.")

