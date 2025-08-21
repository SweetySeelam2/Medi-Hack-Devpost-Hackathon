# HeartRisk Assist — Medi-Hack 2025
**Calibrated, fair, privacy-first cardiac risk triage for rapid decision support**

> Tracks: **AI for Diagnostics** + **Healthcare Operations** (bias/fairness monitoring) + **Mental Health/Privacy** (trust-first framing)

---

## ✨ Why this matters
Emergency departments and primary-care clinics face crowded triage lines and inconsistent risk assessments. Even a small lift in early risk stratification can shorten time-to-care and reduce avoidable downstream testing. **HeartRisk Assist** provides a calibrated probability of cardiac risk, equity slice metrics, and per-patient explanations—*in one lightweight Streamlit app*.

---

## 🔎 What it does
- **Predicts** a **calibrated probability** of “heart disease present” from 13 routine features (UCI Heart dataset).
- **Explains** each prediction (SHAP or coefficient/importance fallback) so clinicians can see **what pushed risk up/down**.
- **Monitors fairness** with slice AUCs by **sex**, **age buckets**, and **chest-pain type**.
- **Shows model quality** (ROC, PR, calibration) and a 95% CI for AUC.
- **Respects privacy**: uses a public de-identified dataset; no PHI; runs locally or on your own server.

---

## 🧠 Key results (test set)

- **AUC:** 0.878  
- **AUPRC:** 0.886  
- **Brier score:** 0.144  
- **AUC 95% CI:** [0.789, 0.949]  
- **Slice AUCs:**  
  - Sex: female 0.846, male 0.871  
  - Age buckets: 45–60 → 0.878; >60 → 0.752  
  - Chest pain type (cp=0): 0.849  
These align with the plots saved under `artifacts/roc.png`, `artifacts/pr.png`, `artifacts/reliability.png`, and `artifacts/shap_summary.png`. :contentReference[oaicite:1]{index=1}

> **Interpretation:** The model is useful (AUC~0.88), reasonably precise over recall (AUPRC~0.89), and *calibration is decent* after isotonic fitting (Brier~0.14). Older patients (>60) show lower discrimination—flagged as an equity risk for threshold audits.

---

## 🗂 Dataset
- **Source:** UCI Heart Disease (public, de-identified).  
- **Rows used:** 302 (deduplicated feature rows from the 1025-row CSV).  
- **Target:** `target` (1 = disease present).  
- **Features:** `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`.

> Note: This small, classic dataset is **appropriate for a hackathon prototype** and transparency demo. For real deployment, retrain on site-specific EHR cohorts (e.g., MIMIC-IV, PhysioNet registries), with IRB and clinical governance.

---

## 🏗️ Modeling
- **Pipeline:** `ColumnTransformer(StandardScaler on numeric + OneHotEncoder on categoricals)` → **LogReg** or **RandomForest** candidate → **isotonic calibration** (`CalibratedClassifierCV` prefit on a validation fold).
- **Model selection:** 5-fold stratified CV by ROC-AUC; best candidate calibrated on a held-out fold; final test on 20% split.
- **Explanations:**  
  - RF → `shap.TreeExplainer` (raw output)  
  - LR → `shap.LinearExplainer`  
  - Robust fallbacks to coefficients/importances if SHAP unavailable.

---

## 🖥️ Streamlit App features
- **Triage (Diagnostics):** sliders/inputs + **calibrated probability** + thresholded **risk band**.
- **Explanations:** top contributing features for the **last prediction**.
- **Fairness (Ops):** JSON of **slice AUCs** for simple bias monitoring.
- **Model Quality:** ROC, PR, reliability plots; **AUC 95% CI**.
- **Batch Scoring:** upload CSV → scores + downloadable results.
- **Sample data:** one-click **Load Sample (High risk)**; background sample also used for SHAP.

---

## 📦 Repo structure
├─ app.py
├─ artifacts/
│ ├─ model.pkl
│ ├─ metrics.json
│ ├─ roc.png · pr.png · reliability.png · shap_summary.png
├─ data/
│ └─ heart.csv
├─ notebooks/
│ └─ training_notebook.ipynb # the notebook you ran
├─ train.py # script version (optional)
├─ requirements.txt
├─ README.md
└─ LICENSE

---

## 🏃 How to run

**1) Setup**
```bash
# Windows (PowerShell)
python -m venv medihack.venv
medihack.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

**2) (Optional) Re-train to regenerate artifacts**

Run the notebook in notebooks/ or:

python train.py


This writes artifacts/model.pkl, artifacts/metrics.json, and plots.

**3) Launch the app**
streamlit run app.py
```

**If you use Jupyter, register the venv:**

> python -m ipykernel install --user --name=medihack --display-name "Python (medihack)"

---

## 📌 Project Links

- Github Repo: https://github.com/SweetySeelam2/Medi-Hack-Devpost-Hackathon

- Streamlit App: 

---

## 🧪 Interpreting the outputs

- Probability is calibrated (reliability curve). Adjust Low/Medium/High thresholds in the sidebar to align with site-specific prevalence and clinical policy.

- Equity: compare slice AUCs; if gaps are large, consider reweighting, collecting under-represented cohorts, or cohort-specific thresholds.

- Explanations: SHAP bars show direction & magnitude of each feature’s contribution for the current case.

---

## 📈 Business impact (order-of-magnitude estimate)

Assume a clinic triages 2,000 patients/month with 15% suspected cardiac cases, average $600 downstream testing cost per suspected case. A calibrated triage that reduces unnecessary follow-up by 5–8% while keeping sensitivity high could save $9k–$14k/month and free clinician time without harming recall (illustrative; verify locally with AB testing and governance).

---

## 🧭 Where this fits

- Health systems: Kaiser Permanente, HCA Healthcare, Cleveland Clinic, Mayo Clinic.

- Payers/providers: UnitedHealth/Optum, Humana, CVS Health (Aetna).

- Digital-health triage/telehealth: included as a transparent, lightweight risk assistant.

---

## ⚠️ Clinical & ethical notes

- Prototype on a small public dataset; not a medical device.

- Requires institutional review, monitoring, and drift audits before clinical use.

- Use site-specific data, add decision thresholds approved by clinical leadership, and run post-deployment bias & safety monitoring.

---

## 📄 License

MIT (c) 2025 Sweety Seelam — see LICENSE

---

## 👤 Author

Sweety Seelam | Business Analyst | Aspiring Data Scientist
- GitHub: https://github.com/SweetySeelam2
- LinkedIn: https://www.linkedin.com/in/sweetyrao670/
- Medium: https://medium.com/@sweetyseelam