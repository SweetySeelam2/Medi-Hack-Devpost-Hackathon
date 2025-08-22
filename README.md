[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medi-hack-devpost-hackathon.streamlit.app/)

---

# HeartRisk Assist — Medi-Hack 2025
**Calibrated, fair, privacy-first cardiac risk triage for rapid decision support**

> **Primary track:** AI for Diagnostics  
> Also relevant to: Healthcare Operations (calibration & fairness), Privacy & Trust

---

## Business Challenge

Clinics must prioritize suspected-cardiac patients when resources and time are limited. The goal is not to diagnose in the app; it’s to triage by producing an accurate, calibrated probability of disease so staff can fast-track high-risk, review medium-risk, and safely monitor low-risk patients.

---

## Blurb For Judges

- Trained a Random Forest (with Logistic Regression as a baseline) and calibrated it with isotonic fitting to predict the probability of heart disease presence. 
- On test data it achieves AUC ≈ 0.878, AUPRC ≈ 0.886, and shows good calibration. 
- Then converted the probability into Low/Medium/High action bands (demo cutoffs 7% / 35%) to support triage, not diagnosis. 
- Explanations (feature contributions) make decisions auditable; fairness slices flag age >60 for monitoring. 
- In a 2,000-visit/month clinic, trimming unnecessary follow-ups by 5–8% at maintained sensitivity saves about $9k–$14k/month. 
- This is a calibrated, transparent triage tool, not a diagnostic device.

---

## Target and prediction

- The dataset label is target where 1 = disease present, 0 = no disease (in the source data).

- The model predicts a calibrated probability
    p = P(disease present | inputs)
    not a hard yes/no.

- The app then maps p → action bands using your demo policy:
    Low < 7%, Medium 7–35%, High ≥ 35%.
    This supports operational decisions (who to fast-track) while avoiding the claim of a clinical diagnosis.

> If a binary decision is required, a single threshold can be applied (e.g., cost-based or capacity-based). You intentionally do not show a “disease: yes/no” label in the UI to avoid over-claiming diagnosis; you show probability + bands for triage.

---

## 🔎 What it does
- **Predicts** a **calibrated probability** that **heart disease is present** from 13 routine inputs (see Dataset).
- **Explains** each prediction (SHAP or robust fallbacks) so I can show **what pushed risk up or down**.
- **Monitors fairness** with slice AUCs by **sex**, **age buckets**, and **chest-pain type**.
- **Shows model quality** (ROC, PR, calibration) with a **95% CI for AUC**.
- **Respects privacy**: public, de-identified data; no PHI; runs locally or on our own server.

---

## 💡 Why this solves the challenge

- Calibration → trust: a “30%” score means ~30% observed risk (reliability curve near the diagonal), enabling defendable thresholds.
- Action bands (demo cutoffs 7% / 35%) align with clinic capacity and acceptable miss rates.
- Explanations clarify why a case is high/medium/low, supporting safer adoption.
- Fairness slices highlight cohorts needing attention (e.g., age > 60 with lower discrimination).
- Operational value: small reductions in unnecessary follow-ups at fixed sensitivity translate into measurable time and cost savings.

---

## ✨ Why this matters
In real clinics, triage is noisy and resources are tight. Even a small lift in **calibrated** risk stratification can shorten time-to-care and reduce avoidable downstream testing.  
**HeartRisk Assist** gives a **calibrated probability** of cardiac risk, clear **Low/Medium/High bands**, **per-patient explanations**, and simple **fairness** checks—inside one lightweight Streamlit app.

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

## 🧠 Key results (test set)
- **AUC:** 0.878  
- **AUPRC:** 0.886  
- **Brier score:** 0.144  
- **AUC 95% CI:** [0.789, 0.949]  
- **Slice AUCs:**  
  - Sex: female 0.846, male 0.871  
  - Age buckets: 45–60 → 0.878; >60 → 0.752  
  - Chest pain type (cp=0): 0.849

These results are derived from `artifacts/roc.png`, `artifacts/pr.png`, `artifacts/reliability.png`, and `artifacts/shap_summary.png`.

---

## 🧪 Interpreting the Results

- **Overall performance:** Discrimination is strong (AUC ≈ 0.88), precision–recall is solid (AUPRC ≈ 0.89), and post–isotonic calibration is good (Brier ≈ 0.14).

- **Calibration meaning:** A calibrated probability of p% means ~p out of 100 similar patients will have the outcome—this is why thresholds/bands are actionable.

- **Action bands (demo policy):** Low < 7%, Medium 7–35%, High ≥ 35%. In production, set these with clinicians using a principled utility/capacity trade-off (triage support, not diagnosis).

- **Fairness & watchpoints:** The >60 age slice shows lower discrimination (ops watchpoint). If slice AUCs diverge, consider reweighting, targeted data collection, or cohort-specific thresholds, and audit thresholds periodically.

- **Governance:** Track reliability curves and slice metrics over time; recalibrate or retune thresholds if drift appears or capacity/priors change.

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

## 📈 Business impact (order-of-magnitude estimate)

- Assume 2,000 monthly visits, 15% suspected cardiac (≈300 triage cases), $600 average downstream test.

- A calibrated triage that trims unnecessary follow-ups by 5–8% (holding sensitivity) saves $9k–$14k/month and frees clinician time

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

## Summary

- **Built a calibrated triage model:** Random Forest as the primary classifier with Logistic Regression as a baseline; final probabilities calibrated via isotonic regression to make risk scores usable in operations.

- **Auditable decisions:** per-prediction feature contributions make outputs explainable; fairness slices are monitored, with an early watchpoint for patients age > 60.

- **Operational impact (illustrative):** In a 2,000-visit/month clinic, trimming unnecessary follow-ups by 5–8% at maintained sensitivity translates to an estimated $9k–$14k/month in savings (actuals depend on local follow-up costs and workflows).

- **Model quality (test set):** Strong discrimination and reliability with:
    AUC = {auc:.3f}, AUPRC = {aupr:.3f}, Brier = {brier:.3f}, AUC 95% CI = [{lo_ci:.3f}, {hi_ci:.3f}].

- Calibration: good post-isotonic fitting—predicted probabilities align with observed risk, enabling threshold-based actions.

- **Action bands for triage (demo cutoffs):**
    Low: p < 7% → routine care
    Medium: 7–35% → clinician review / follow-ups as needed
    High: > 35% → prioritize assessment and risk management
    (Bands are for workflow support, not diagnosis.)

- **Equity watchpoint:** older patients show lower discrimination (AUC {age_gt60:.3f}) relative to ages 45–60 (AUC {age_45_60:.3f}); this cohort is flagged for ongoing monitoring and potential threshold or workflow adjustments.

***Bottom line: a calibrated, transparent triage tool to support prioritization and resource use—not a diagnostic device.***

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