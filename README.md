<h1 align="center">🔧 Pipeline Corrosion Prediction & Monitoring Dashboard</h1>
<h3 align="center">Final Year Project – Universiti Teknologi PETRONAS (UTP)</h3>

<p align="center">
<b>Author:</b> Muhammad Hanis Afifi Bin Azmi <br>
<b>Supervisor:</b> Ts. Faizal B Ahmad Fadzil <br>
<b>Course:</b> Final Year Project (FYP) <br>
</p>

<hr>

<h2>📌 Project Overview</h2>
<p>
This project presents a complete <b>Machine Learning + Deep Learning corrosion prediction system</b> designed to assist 
pipeline engineers in monitoring, predicting, and visualizing corrosion severity across multiple regions in Malaysia.

The dashboard integrates three ML models:
<ul>
  <li><b>Deep Learning Regression (TensorFlow)</b></li>
  <li><b>Random Forest Regressor</b></li>
  <li><b>XGBoost Regressor</b></li>
</ul>

These models are combined into a <b>Reinforced Ensemble</b> to significantly improve prediction stability and accuracy.

The final predictions are displayed through an interactive 
<b>Streamlit dashboard</b> that enables engineers to visualize:
<ul>
  <li>Real-time corrosion rate estimates</li>
  <li>Severity classification (Low / Medium / High)</li>
  <li>Pipeline remaining life estimation</li>
  <li>Material & environment correlations</li>
  <li>Model comparison and performance analytics</li>
</ul>
</p>

<hr>

<h2>📂 Repository Structure</h2>

<pre>
📁 FYP_Corrosion_V1/
│── 📜 corrosion_predictor_app.py     → Main Streamlit dashboard
│── 📜 final_corrosion_model.keras     → Trained Deep Learning model
│── 📜 preprocessor_corrosion.joblib   → Data preprocessing pipeline
│── 📜 rf_model.joblib                 → Random Forest model  
│── 📜 xgb_model.json                  → XGBoost model
│── 📂 dataset/
│     └── cleaned_corrosion_regression_data.csv
│── 📂 predictions/                    → Saved output CSV from user uploads
│── 📂 images/                         → Supporting images / UTP logo
│── 📜 requirements.txt
│── 📜 README.md
</pre>

<hr>

<h2>🚀 How Engineers Use This Dashboard</h2>

<h3>1️⃣ Predict corrosion rate for any pipeline</h3>
Engineers can upload a CSV containing pipeline chemical/environmental parameters:
<ul>
  <li>Temperature (°C)</li>
  <li>CO₂ concentration (%)</li>
  <li>Environment type</li>
  <li>Material family</li>
</ul>

The system automatically:
<ul>
  <li>Runs preprocessing</li>
  <li>Predicts corrosion rates using 3 models</li>
  <li>Applies ensemble averaging</li>
</ul>

<h3>2️⃣ Visualize pipeline condition by region</h3>
Pipelines are grouped into:
<ul>
  <li>Peninsular Malaysia</li>
  <li>Sabah</li>
  <li>Sarawak</li>
</ul>

Each pipeline is displayed as a <b>color-coded indicator</b> representing its severity:
<ul>
  <li>🟢 Low corrosion</li>
  <li>🟡 Moderate corrosion</li>
  <li>🔴 High corrosion</li>
</ul>

<h3>3️⃣ Estimate Remaining Life (RLA)</h3>
Engineers can input:
<ul>
  <li>Current wall thickness</li>
  <li>Minimum allowable thickness</li>
</ul>

The dashboard estimates:
<b>
Remaining service life = (T_current − T_min) / Effective corrosion rate
</b>

<h3>4️⃣ Perform Corrosion Trend Analysis</h3>
Engineers can explore:
<ul>
  <li>Temperature vs corrosion rate</li>
  <li>Material-dependent corrosion behavior</li>
  <li>Correlation heatmaps</li>
  <li>Model validation scatter plots</li>
</ul>

<hr>

<h2>🌟 Key Features & Engineering Impact</h2>

<h3>✔ 1. Improves decision-making for pipeline maintenance</h3>
The system helps engineers decide:
<ul>
  <li>Which pipelines require inspection</li>
  <li>When shutdown or replacement is needed</li>
  <li>Which material performs best in corrosive environments</li>
</ul>

<h3>✔ 2. Enhances safety & reliability</h3>
Predicting corrosion early prevents:
<ul>
  <li>Pipeline leaks</li>
  <li>Environmental spills</li>
  <li>Unplanned shutdowns</li>
</ul>

<h3>✔ 3. Cost reduction</h3>
Intelligent prediction reduces:
<ul>
  <li>Inspection frequency</li>
  <li>Maintenance cost</li>
  <li>Material wastage</li>
</ul>

<h3>✔ 4. Practical for real-world application</h3>
The dashboard is:
<ul>
  <li>Lightweight & fast</li>
  <li>Can run on cloud or local PC</li>
  <li>Requires only simple CSV inputs</li>
</ul>

<h3>✔ 5. High prediction accuracy (Validated)</h3>
<ul>
  <li><b>Deep Learning R² = 0.99</b></li>
  <li><b>Random Forest R² = 0.87</b></li>
  <li><b>XGBoost R² = 0.85</b></li>
</ul>

The ensemble gives stable, reliable predictions for operational use.

<hr>

<h2>🛠️ Installation Guide</h2>

<h3>Requirements</h3>
<pre>
Python 3.10+
TensorFlow 2.9
scikit-learn
xgboost
streamlit
plotly
pandas, numpy, seaborn
</pre>

<h3>Steps</h3>

<pre>
1. Clone repository:
   git clone https://github.com/hanis-ctrlcv/FYP_Corrosion_V1.git
   cd FYP_Corrosion_V1

2. Install dependencies:
   pip install -r requirements.txt

3. Run dashboard:
   streamlit run corrosion_predictor_app.py
</pre>

<hr>

<h2>📞 Contact</h2>
<p>
For academic or industrial collaboration, feel free to reach out:
<br>
<b>Email:</b> <i>azmi.hanisafifi@gmail.com</i>
</p>

<hr>

<h2 align="center">🎉 Thank You for Exploring the Project!</h2>
<p align="center">This system aims to support safer, smarter, and more efficient pipeline operations.</p>
