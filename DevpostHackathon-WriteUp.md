## Project Title
**HeartRisk Assist — Calibrated, Fair, Privacy-First Cardiac Triage**

### Elevator pitch (≤30s)
I built **HeartRisk Assist** to give clinicians a **calibrated probability** of cardiac risk, with **fairness slice metrics** and **point-of-care explanations** in one simple Streamlit app. It’s fast, transparent, and privacy-respecting—ideal for triage support and operations dashboards.

### Track(s)
- **Primary:** AI for Diagnostics  
- **Also fits:** Healthcare Operations (calibration & fairness), Privacy & Trust

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

> If a binary decision is required, apply a single threshold (e.g., capacity- or cost-based). The UI intentionally shows **probability + band** instead of “disease: yes/no”.

---

## Why this solves the challenge

- **Calibration → trust:** a “30%” score means ~30% observed risk, enabling **defendable thresholds**.  
- **Action bands** (**7% / 35%** demo) align with **clinic capacity** and acceptable **miss tolerance**.  
- **Explanations** clarify *why* a case is high/medium/low → safer adoption.  
- **Fairness slices** surface cohorts needing attention (e.g., **>60** years shows lower discrimination).  
- **Operational value:** small reductions in unnecessary follow-ups at fixed sensitivity translate to measur

---

## What it does
- Predicts a **calibrated probability** that heart disease is present from **13 common inputs**.  
- Explains each prediction (**SHAP** or fallbacks) so it’s clear **why** risk went up or down.  
- Surfaces **equity metrics** (slice AUCs by **sex**, **age bucket**, **chest-pain type**).  
- Shows **ROC/PR**, **reliability (calibration)**, and **AUC 95% CI** for trust and governance.  
- Batch-scores CSVs; one-click **Load Sample (High risk)** for demo.

---

## Why it matters
Triage lines are crowded and risk estimates vary. **Calibration** turns scores into **probabilities** clinicians can act on. Even modest improvements in calibrated risk stratification reduce unnecessary testing and speed up care. The **calibration + fairness** framing makes performance legible to both clinicians and operations leaders.

---

## How I built it
- **Data:** Kaggle Heart Disease dataset (public, de-identified), **302 deduplicated rows** to avoid leakage/duplicates.
- **Modeling:** `ColumnTransformer` (scale numeric + one-hot categoricals) → **Random Forest (final)** and **Logistic Regression (baseline)** → **isotonic calibration** on a held-out fold → test evaluation.
- **Explainability:** `shap.TreeExplainer` for RF, `shap.LinearExplainer` for LR, with robust fallbacks if SHAP is not available.
- **App:** Streamlit multi-page UX (Triage, Explanations, Fairness, Model Quality, Data Explorer, Batch Scoring).  
- **Threshold policy:** demo defaults **Low < 7%**, **High ≥ 35%** with a **principled method** documented for choosing site-specific cutoffs.

---

## Results (test set)
- **AUC:** 0.878 | **AUPRC:** 0.886 | **Brier:** 0.144 | **AUC 95% CI:** [0.789, 0.949]  
- **Slice AUCs:** sex (F 0.846, M 0.871); age 45–60 (0.878), >60 (0.752); cp=0 (0.849).  
Plots (ROC, PR, calibration, SHAP) are in `artifacts/`.

**Takeaway:** Useful discrimination and good calibration. The **>60** cohort shows lower discrimination—an action point for threshold audits and more data.

---

## Key feature findings (what drives risk in this model)
*Associations captured by the calibrated Random Forest with SHAP—**not** causal claims.*

- **oldpeak (ST depression):** higher values **↑ risk sharply**; strongest driver.  
- **thal (perfusion category):** **reversible (3)** or **fixed defect (1)** → **↑ risk** vs **normal (2)**.  
- **ca (vessels via fluoroscopy):** **2–3 vessels** → **↑ risk** vs **0–1**.  
- **slope (ST segment):** **downsloping (2)** → **↑ risk**; **upsloping (0)** → **↓ risk**; **flat (1)** between.  
- **exang:** **1 (yes)** → **↑ risk** vs **0 (no)**.  
- **thalach (max HR):** **lower** thalach → **↑ risk** (higher fitness protective).  
- **age:** risk **increases** with age.  
- **restecg:** **abnormal (1/2)** → **↑ risk** vs **normal (0)**.  
- **trestbps / chol:** smaller, positive contributions at higher values.  
- **sex:** **male (1)** slightly **higher risk** than **female (0)** in this dataset.  
- **cp (chest-pain type):** **asymptomatic (3)** highest; **typical/atypical (0/1)** lower; **non-anginal (2)** between.

**Takeaway:** The most influential risk-increasing drivers are higher ST depression (oldpeak), abnormal thal, more fluoroscopy-visible vessels (ca), downsloping ST slope, exercise-induced angina, older age, and rest ECG abnormalities. Higher max heart rate (thalach) is protective.

- Encoding key (Heart dataset conventions):
· cp: 0=typical, 1=atypical, 2=non-anginal, 3=asymptomatic 
· slope: 0=up, 1=flat, 2=down 
· thal: 1=fixed defect, 2=normal, 3=reversible defect 
· exang: 0=no, 1=yes 
· restecg: 0=normal, 1=ST-T abn., 2=LVH 
· sex: 0=female, 1=male

***Note: These are predictive associations from SHAP on the calibrated Random Forest; they are not causal claims and may shift with cohort or data drift.***

---

## Challenges
- **Explaining calibrated models:** Unwrapped `CalibratedClassifierCV` to explain the underlying estimator cleanly.
- **Tiny dataset / overfit risk:** Constrained RF search, held-out calibration, bootstrap CI for AUC.
- **Windows long-path/wheels:** Handled via venv hygiene and path settings.

---

## What I’m proud of
- A **cohesive, demo-ready** workflow that blends diagnostics, fairness, and trust.  
- Fully reproducible artifacts and a small app that anyone can run.

---

## What’s next
- Retrain on site EHR cohorts (e.g., MIMIC-IV/PhysioNet) with governance.
- Add **threshold-setting tools** (capacity/PPV, miss-rate/NPV, decision curves).
- Run prospective A/B tests and post-deployment drift/bias monitoring.

---

## Try it
- **Code:** https://github.com/SweetySeelam2/Medi-Hack-Devpost-Hackathon  
- **Run locally:** `streamlit run app.py` (see README for steps).  
- **Streamlit App:** https://medi-hack-devpost-hackathon.streamlit.app/  
- **Demo video:** 

---

## Team
**Solo: Sweety Seelam**

---

## Built with
Python, scikit-learn, SHAP, pandas, numpy, matplotlib, Streamlit