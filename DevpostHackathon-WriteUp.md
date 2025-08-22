## Project Title
**HeartRisk Assist — Calibrated, Fair, Privacy-First Cardiac Triage**

### Elevator pitch (≤30s)
I built **HeartRisk Assist** to give clinicians a **calibrated probability** of cardiac risk, with **fairness slice metrics** and **point-of-care explanations** in one simple Streamlit app. It’s fast, transparent, and privacy-respecting—ideal for triage support and operations dashboards.

### Track(s)
- **Primary:** AI for Diagnostics  
- **Also fits:** Healthcare Operations (calibration & fairness), Privacy & Trust

---

## What it does
- Predicts a **calibrated probability** that heart disease is present, from 13 common inputs.
- Explains each prediction (SHAP or fallbacks) so it’s clear **why** the risk went up or down.
- Surfaces **equity metrics** (slice AUCs by sex, age bucket, chest-pain type).
- Shows **ROC/PR**, **reliability**, and **AUC 95% CI** for trust and governance.
- Batch-scores CSVs; one-click **Load Sample (High risk)** for demo.

---

## Why it matters
Triage lines are crowded and risk estimates vary. Calibration turns scores into **probabilities** clinicians can act on. Even modest improvements in calibrated risk stratification reduce unnecessary testing and speed up care. The **calibration + fairness** framing makes performance legible to both clinicians and operations leaders.

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

## Challenges
- **Explaining calibrated models:** I unwrapped `CalibratedClassifierCV` to explain the underlying estimator cleanly.
- **Tiny dataset / overfit risk:** constrained RF search, held-out calibration, bootstrap CI for AUC.
- **Windows long-path/wheels:** handled via venv hygiene and path settings.

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
- **Demo video:** (Attached video demo capturing Page 1 → Triage → Explanations → Fairness → Model Quality → Impact)

---

## Team
**Solo: Sweety Seelam**

---

## Built with
Python, scikit-learn, SHAP, pandas, numpy, matplotlib, Streamlit