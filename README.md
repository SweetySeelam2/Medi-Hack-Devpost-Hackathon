[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medi-hack-devpost-hackathon.streamlit.app/)

---

# HeartRisk Assist — Medi-Hack 2025
**Calibrated, fair, privacy-first cardiac risk triage for rapid decision support**

> **Primary track:** AI for Diagnostics  
> Also relevant to: Healthcare Operations (calibration & fairness), Privacy & Trust

---

## ✨ Why this matters
In real clinics, triage is noisy and resources are tight. Even a small lift in **calibrated** risk stratification can shorten time-to-care and reduce avoidable downstream testing.  
**HeartRisk Assist** gives a **calibrated probability** of cardiac risk, clear **Low/Medium/High bands**, **per-patient explanations**, and simple **fairness** checks—inside one lightweight Streamlit app.

---

## 🔎 What it does
- **Predicts** a **calibrated probability** that **heart disease is present** from 13 routine inputs (see Dataset).
- **Explains** each prediction (SHAP or robust fallbacks) so I can show **what pushed risk up or down**.
- **Monitors fairness** with slice AUCs by **sex**, **age buckets**, and **chest-pain type**.
- **Shows model quality** (ROC, PR, calibration) with a **95% CI for AUC**.
- **Respects privacy**: public, de-identified data; no PHI; runs locally or on your own server.

---

## 🧭 How thresholds were chosen (7% / 35%)
These are **demo defaults**, not clinical truth. Encoded a common operations policy: keep **High** a smaller subset to **fast-track**, and make **Low** truly low risk.

**Principled selection you can justify**
1) **Decide operational targets first**
   - **Capacity target** for fast-track (e.g., 15–25% of cases).
   - **Miss tolerance** in **Low** (e.g., ≤3–5 events per 100 Low-band patients).
   - Optional floors: **PPV** for High, **NPV** for Low.

2) **Use calibration to place two cutoffs**
   - **High (t_high)** — choose the smallest *t* so only your top **C%** of cases have *p ≥ t* **or** so PPV among *p ≥ t* meets your target.  
   - **Low (t_low)** — raise *t* from 0 until expected events among *p < t* stay within your allowed misses per 100 Low patients.

   On this dataset, these rules commonly land around **0.05–0.10** for Low and **0.30–0.40** for High → my demo defaults **0.07 / 0.35**.

3) **(Two-way option).** If you want a single “escalate vs not” cutoff, use the cost-ratio formula:
`t* = C_FP / (C_FP + C_FN)`.  
Example: if a miss (FN) is about **10×** a false alarm (FP), then `t* ≈ 1/11 ≈ 0.09`.

*In production, replace the demos with site-specific thresholds derived from your validation set and clinical policy.*                              

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

These align with `artifacts/roc.png`, `artifacts/pr.png`, `artifacts/reliability.png`, and `artifacts/shap_summary.png`.

> **Interpretation:** The model is useful (AUC≈0.88), reasonably precise over recall (AUPRC≈0.89), and **probabilities are trustworthy** after isotonic fitting (Brier≈0.14). Older patients (>60 age slice) shows lower discrimination — an ops watchpoint, flagged for threshold audits and more data collection.

---

## 🗂 Dataset
- **Source:** Kaggle Heart Disease dataset (public, de-identified): https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset  
- **Rows used:** 302 (**deduplicated feature rows** from the original CSV to avoid leakage/duplicates).  
- **Target:** `target` (1 = disease present, 0 = no disease).  
  The app predicts a calibrated probability for target = 1 and then maps it to action bands.
- **Features:** `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`.  
- **License/Use:** Educational research/prototyping.

> This small, classic dataset is ideal for a hackathon prototype and transparency demo. For deployment, retrain on site-specific EHR cohorts (e.g., MIMIC-IV/PhysioNet) with IRB and clinical governance.

---

## 🏗️ Modeling
- **Pipeline:** `ColumnTransformer` (StandardScaler on numeric + OneHotEncoder on categoricals) 
  → candidate model (**LogisticRegression** or **RandomForest**) → **isotonic calibration** with `CalibratedClassifierCV` on a held-out validation fold.
- **Selection:** 5-fold stratified CV by ROC-AUC; best candidate is calibrated; final metrics on a 20% test split.
- **Explainability:**  
  - RF → `shap.TreeExplainer` (raw output)  
  - LR → `shap.LinearExplainer`  
  - Robust fallbacks (coefficients/importances or directional ablations) if SHAP is unavailable.

---

## 🖥️ Streamlit app features
- **Triage (Diagnostics):** Enter inputs or click Load sample (High risk) → get calibrated probability and a band using the fixed demo policy (Low < 7%, Medium 7–35%, High ≥ 35%).
  Interpretation text explains “per-100 patients” to make calibration tangible.
- **Explanations:** Vertical bars show what increased (above zero) or reduced (below zero) risk for the current case.
- **Fairness (Ops):** Quick slice AUCs so I can spot cohort gaps.
- **Model Quality:** ROC, PR, calibration curve, and AUC CI (95% CI) to judge utility and trust.
- **Batch Scoring:** Upload CSV → download probabilities + bands for many rows at once.
- **Data Explorer:** Browse the 302-row training sample and push any row to Triage.

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
│ └─ train.ipynb                                                                                             
├─ train.py # script version (optional)                                                                                     
├─ requirements.txt                                                                             
├─ README.md                                                                                     
└─ LICENSE                                                                                        

---

## 🏃 How to run

```bash
1) Setup

# Windows (PowerShell)
python -m venv medihack.venv
medihack.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

2) (Optional) Re-train to regenerate artifacts

Run the notebook in notebooks/ or:

python train.py

This writes artifacts/model.pkl, artifacts/metrics.json, and plots.

3) Launch the app
streamlit run app.py
```

**If you use Jupyter, register the venv:**
```bash
python -m ipykernel install --user --name=medihack --display-name "Python (medihack)"
```

---

## 📌 Project Links

- Github Repo: https://github.com/SweetySeelam2/Medi-Hack-Devpost-Hackathon

- Streamlit App: https://medi-hack-devpost-hackathon.streamlit.app/

- Video Demo: 
---

## 🧪 Interpreting the outputs

- **Calibrated probability** means “p% ≈ p out of 100 similar patients.” That’s why bands are meaningful.

- **Bands (demo policy):** Low < 7%, Medium 7–35%, High ≥ 35%. In production, set these with clinicians using the principled method above.

- **Fairness:** If slice AUCs diverge (e.g., >60 age bucket), consider reweighting, more data, or cohort-specific thresholds.

---

## 📈 Business impact (order-of-magnitude estimate)

- Assume 2,000 monthly visits, 15% suspected cardiac (≈300 triage cases), $600 average downstream test.

- A calibrated triage that trims unnecessary follow-ups by 5–8% (holding sensitivity) saves $9k–$14.4k/month and frees clinician time

---

## 🧭 Where this fits

- Health systems: Kaiser Permanente, HCA Healthcare, Cleveland Clinic, Mayo Clinic.

- Payers/providers: UnitedHealth/Optum, Humana, CVS Health (Aetna).

- Digital-health triage/telehealth: included as a transparent, lightweight risk assistant.

---

## ⚠️ Clinical & ethical notes

- Prototype on a small public dataset; not a medical device.

- Requires institutional review, monitoring, and drift audits before clinical use.

- Use site-specific data, add approved decision thresholds, and run post-deployment bias & safety monitoring.

---

## 🧰 Environment & reproducibility

- Python 3.10; scikit-learn ≥1.4; SHAP ≥0.44; Streamlit ≥1.33

- Global seed: RANDOM_STATE = 42

- Artifacts stored in artifacts/ with metrics for traceability.

---

## 🧾 Original work & AI attribution

All code and analysis were created during the Medi-Hack hackathon window.                                        
Open-source tools (scikit-learn, SHAP, Streamlit) were used.                                   
AI assistance was used for code reviews and documentation polish; final logic and validation are my own.                          

---

## Team
Solo: **Sweety Seelam**

---

## 👤 Author

**Sweety Seelam | Business Analyst | Aspiring Data Scientist**
- GitHub: https://github.com/SweetySeelam2
- LinkedIn: https://www.linkedin.com/in/sweetyrao670/
- Medium: https://medium.com/@sweetyseelam

---

## 📄 License

MIT (c) 2025 Sweety Seelam — see LICENSE