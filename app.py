# app.py — HeartRisk Assist (Medi-Hack 2025)
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

# ---------- Page setup ----------
st.set_page_config(page_title="HeartRisk Assist — Medi-Hack Edition", page_icon="❤️", layout="wide")
st.markdown("## HeartRisk Assist — Calibrated, Fair, Privacy-First Cardiac Risk Triage (Medi-Hack 2025)")
st.info("Built during Medi-Hack (Aug 18–23, 2025). Educational prototype — **not medical advice**. No PHI used.")

# ---------- Paths ----------
ART = Path("artifacts")
DATA_PATH = Path("data/heart.csv")

# ---------- Cached loaders ----------
@st.cache_resource
def load_bundle():
    return joblib.load(ART / "model.pkl")

@st.cache_data
def load_metrics():
    with open(ART / "metrics.json") as f:
        return json.load(f)

@st.cache_data
def load_bg(features):
    if DATA_PATH.exists():
        try:
            return pd.read_csv(DATA_PATH)[features].copy()
        except Exception:
            return None
    return None

# ---------- Load artifacts ----------
BUNDLE   = load_bundle()                     # saved by train script/notebook
MODEL    = BUNDLE["model"]                   # CalibratedClassifierCV wrapping a Pipeline(pre, clf)
FEATURES = BUNDLE["features"]
PREPROC  = BUNDLE.get("preprocessor", None)  # optional; used only as a fallback

METRICS = load_metrics()
DF_BG   = load_bg(FEATURES)

# ---------- Dictionaries & samples ----------
CATS = {
    "sex":    {0: "Female (0)", 1: "Male (1)"},
    "cp":     {0: "Typical angina (0)", 1: "Atypical angina (1)", 2: "Non-anginal pain (2)", 3: "Asymptomatic (3)"},
    "fbs":    {0: "Fasting blood sugar ≤ 120 mg/dL (0)", 1: "Fasting blood sugar > 120 mg/dL (1)"},
    "restecg":{0: "Normal (0)", 1: "ST-T wave abnormality (1)", 2: "LV hypertrophy (2)"},
    "exang":  {0: "No (0)", 1: "Yes (1)"},
    "slope":  {0: "Upsloping (0)", 1: "Flat (1)", 2: "Downsloping (2)"},
    "ca":     {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal":   {0: "Unknown (0)", 1: "Fixed defect (1)", 2: "Normal (2)", 3: "Reversible defect (3)"},
}
NUM_META = {
    "age":      {"label": "Age (years)",               "min": 18, "max": 95,  "step": 1.0},
    "trestbps": {"label": "Resting BP (mm Hg)",        "min": 80, "max": 220, "step": 1.0},
    "chol":     {"label": "Serum cholesterol (mg/dL)", "min": 100,"max": 600, "step": 1.0},
    "thalach":  {"label": "Max heart rate (bpm)",      "min": 60, "max": 230, "step": 1.0},
    "oldpeak":  {"label": "ST depression (mm)",        "min": 0.0,"max": 7.0, "step": 0.1},
}
SAMPLE = {  # canonical higher-risk example
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0,
    "thalach": 150, "exang": 1, "oldpeak": 2.3, "slope": 0, "ca": 2, "thal": 3
}

# ---------- Helpers ----------
def license_footer():
    st.markdown(
        "<hr/><div style='font-size:12px'>MIT License — (c) 2025 Sweety Seelam. "
        "This tool is an educational triage prototype, not medical advice.</div>",
        unsafe_allow_html=True
    )

def unwrap_to_fitted_pipeline(model):
    """
    Return (pre, clf, pipe) where `pre` is the FITTED preprocessor and `clf` is the fitted classifier.
    Works when the model is CalibratedClassifierCV(Pipeline(pre, clf)).
    Falls back to PREPROC fitted on background data for explanations only.
    """
    est = model
    # Unwrap calibrated model to the underlying estimator
    if isinstance(model, CalibratedClassifierCV):
        # Prefer the actually-used estimator stored in calibrated_classifiers_
        if getattr(model, "calibrated_classifiers_", None):
            cc = model.calibrated_classifiers_[0]
            est = getattr(cc, "estimator", getattr(cc, "base_estimator", model))
        else:
            est = getattr(model, "estimator", getattr(model, "base_estimator", model))

    if isinstance(est, Pipeline) and hasattr(est, "named_steps"):
        pre = est.named_steps.get("pre", None)
        clf = est.named_steps.get("clf", None)
        if pre is None or clf is None:
            raise RuntimeError("Pipeline missing expected 'pre'/'clf' steps.")
        return pre, clf, est

    # Fallback path: use saved PREPROC for explanation-only transforms
    pre = PREPROC
    if pre is None:
        raise RuntimeError("No preprocessor available for explanations.")
    # If not fitted, fit on background data (harmless for explanation only)
    if not hasattr(pre, "transformers_"):
        if DF_BG is None or DF_BG.empty:
            raise RuntimeError("Preprocessor is not fitted and no background data is available.")
        pre.fit(DF_BG[FEATURES])
    clf = est
    return pre, clf, None

def risk_band(prob, lo, hi):
    if prob < lo: return "Low"
    if prob < hi: return "Medium"
    return "High"

def explain_single(x_df):
    """
    Returns (method_str, pd.DataFrame of top contributions) for a single-row input.
    Tries SHAP first; falls back to coefficients/feature importances.
    """
    pre, clf, _ = unwrap_to_fitted_pipeline(MODEL)

    # Transform into model feature space
    Xt = pre.transform(x_df[FEATURES])
    feat_names = list(getattr(pre, "get_feature_names_out", lambda *_: FEATURES)(FEATURES))
    row_dense = Xt.toarray()[0] if hasattr(Xt, "toarray") else np.asarray(Xt)[0]

    # Try SHAP
    try:
        import shap
        bg = DF_BG.sample(min(len(DF_BG), 200), random_state=42) if DF_BG is not None else x_df[FEATURES].copy()
        Xbg = pre.transform(bg)
        if hasattr(Xbg, "toarray"): Xbg = Xbg.toarray()
        Xtd = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)

        if hasattr(clf, "estimators_"):            # Tree models
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(Xtd)        # list for binary: [neg, pos]
            vals = (np.asarray(sv[1])[0] if isinstance(sv, list) else np.asarray(sv)[0])
            method = "SHAP (TreeExplainer)"
        else:                                      # Linear models
            explainer = shap.LinearExplainer(clf, Xbg)
            try:
                exp = explainer(Xtd)
                vals = np.asarray(getattr(exp, "values", exp))[0]
            except Exception:
                vals = np.asarray(explainer.shap_values(Xtd))[0]
            method = "SHAP (LinearExplainer)"

        contrib = (pd.DataFrame({"feature": feat_names, "contribution": vals})
                   .sort_values("contribution", key=np.abs, ascending=False)
                   .head(15))
        return method, contrib

    except Exception:
        # Robust fallbacks if SHAP is missing or fails
        if hasattr(clf, "coef_"):
            vals = row_dense * clf.coef_[0]
            method = "Coefficient-based"
        elif hasattr(clf, "feature_importances_"):
            vals = row_dense * clf.feature_importances_
            method = "Importance-based"
        else:
            return "Unavailable", pd.DataFrame(columns=["feature", "contribution"])

        contrib = (pd.DataFrame({"feature": feat_names, "contribution": vals})
                   .sort_values("contribution", key=np.abs, ascending=False)
                   .head(15))
        return method, contrib

def fill_sample():
    for k, v in SAMPLE.items():
        st.session_state[k] = v

def push_row_to_triage(row_dict):
    for k, v in row_dict.items():
        st.session_state[k] = v
    st.session_state["nav"] = "Triage (Diagnostics)"

# ---------- Sidebar controls ----------
st.sidebar.header("Risk Thresholds")
# Sensible defaults for calibrated clinical triage
lo = st.sidebar.slider("Low/Medium cutoff", 0.00, 0.50, 0.07, 0.01)
hi = st.sidebar.slider("Medium/High cutoff", 0.20, 0.90, 0.35, 0.01)
if hi <= lo:
    st.sidebar.error("Medium/High cutoff must be > Low/Medium cutoff.")
st.session_state["lo"] = lo
st.session_state["hi"] = hi

with st.sidebar.expander("What are thresholds?", expanded=False):
    st.markdown(
        f"""
Map a **calibrated probability** into an action band:

- **Low**: p \< {lo:.0%} → routine prevention & monitoring  
- **Medium**: {lo:.0%} ≤ p \< {hi:.0%} → consider non-invasive tests; clinician review  
- **High**: p ≥ {hi:.0%} → prompt clinician evaluation

Tune these to your clinic’s prevalence and risk tolerance.
        """
    )

NAV_OPTIONS = [
    "Triage (Diagnostics)", "Explanations", "Fairness (Ops)",
    "Model Quality", "Batch Scoring", "Data Explorer", "Privacy & Trust"
]
page = st.sidebar.radio("Go to", NAV_OPTIONS, key="nav")

# ---------- Page: TRIAGE ----------
if page == "Triage (Diagnostics)":
    st.subheader("Enter measurements")

    with st.expander("How to use", expanded=False):
        st.markdown("""
1) Enter routine measurements below (or click **Load sample (High risk)**).  
2) Click **Assess risk** → you’ll see a **calibrated probability** and **risk band** driven by the sidebar thresholds.  
3) Open **Explanations** to see which features pushed risk up or down for the last case.
        """)

    tips = {
        "sex": "Biological sex (0=Female, 1=Male).",
        "cp": "Chest pain type: 0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic.",
        "restecg": "Resting ECG: 0=Normal, 1=ST-T abnormality, 2=LV hypertrophy.",
        "exang": "Exercise-induced angina: 0=No, 1=Yes.",
        "slope": "Slope of peak exercise ST segment: 0=Upsloping, 1=Flat, 2=Downsloping.",
        "ca": "# major vessels (0–3) colored by fluoroscopy.",
        "thal": "Thalassemia: 0=Unknown, 1=Fixed defect, 2=Normal, 3=Reversible defect.",
        "thalach": "Maximum heart rate achieved (bpm).",
        "oldpeak": "ST depression induced by exercise relative to rest (mm).",
    }

    # Build the form with friendly widgets
    with st.form("triage_form", clear_on_submit=False):
        cols = st.columns(3)
        vals = {}

        for i, f in enumerate(FEATURES):
            with cols[i % 3]:
                if f in CATS:  # categorical with labels
                    default_code = int(st.session_state.get(f, SAMPLE.get(f, list(CATS[f].keys())[0])))
                    vals[f] = st.selectbox(
                        f, options=list(CATS[f].keys()),
                        index=list(CATS[f].keys()).index(default_code),
                        format_func=lambda k, _f=f: CATS[_f][k],
                        help=tips.get(f, "")
                    )
                else:          # numeric with sensible bounds
                    meta = NUM_META.get(f, {"label": f, "min": 0.0, "max": 9999.0, "step": 1.0})
                    default_val = float(st.session_state.get(f, SAMPLE.get(f, 0.0)))
                    vals[f] = st.number_input(meta["label"] if meta["label"] else f,
                                              value=default_val,
                                              min_value=float(meta["min"]),
                                              max_value=float(meta["max"]),
                                              step=float(meta["step"]),
                                              help=tips.get(f, ""))

        submitted = st.form_submit_button("Assess risk")

    if submitted:
        x = pd.DataFrame([vals])[FEATURES]
        prob = float(MODEL.predict_proba(x)[:, 1])   # pipeline + calibration inside
        band = risk_band(prob, lo, hi)

        st.metric("Calibrated risk", f"{prob:.1%}")
        st.success(
            f"**Risk Band: {band}** — because **{prob:.1%}** "
            f"{'<' if prob < lo else ('between' if prob < hi else '≥')} "
            f"{lo:.0% if prob < lo else (str(int(lo*100))+'% and '+str(int(hi*100))+'%' if prob < hi else str(int(hi*100))+'%')}."
        )
        st.caption("Probabilities are **calibrated**; explanations (next page) reflect the model before calibration.")

    # Use button return (not on_click) so we can rerun safely
    if st.button("Load sample (High risk)", type="secondary",
                 help="Pre-fills fields with a typical higher-risk profile"):
        fill_sample()
        st.rerun()

    license_footer()

# ---------- Page: EXPLANATIONS ----------
elif page == "Explanations":
    st.subheader("Why this decision?")
    st.caption("Local explanation for the most recent inputs. If none submitted yet, we use a representative sample.")

    row = {f: st.session_state.get(f, SAMPLE.get(f, 0)) for f in FEATURES}
    x = pd.DataFrame([row])[FEATURES]
    method, contrib = explain_single(x)
    if contrib.empty:
        st.warning("Explanation currently unavailable for this configuration.")
    else:
        st.write(f"Explanation method: **{method}**")
        st.bar_chart(contrib.set_index("feature")["contribution"])
        st.caption("Positive bars push toward higher risk; negative bars reduce risk.")
    license_footer()

# ---------- Page: FAIRNESS ----------
elif page == "Fairness (Ops)":
    st.subheader("Slice metrics (AUC)")
    st.caption("We report AUC by **sex**, **age buckets** (lt45, 45–60, gt60), and **cp** (chest pain type).")
    st.json({
        "auc_by_sex": METRICS.get("auc_by_sex", {}),
        "auc_by_age_bucket": METRICS.get("auc_by_age_bucket", {}),
        "auc_by_cp": METRICS.get("auc_by_cp", {})
    })
    st.markdown("""
**Operational use:** monitor equity over time. If gaps are large, collect more data in the under-served cohort,
consider reweighting, or tune thresholds by cohort with clinical oversight.
    """)
    license_footer()

# ---------- Page: MODEL QUALITY ----------
elif page == "Model Quality":
    st.subheader("Validation metrics")
    clean = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in METRICS.items()}
    st.write(clean)
    if "auc_ci" in METRICS:
        st.write(f"Test AUC 95% CI: [{METRICS['auc_ci'][0]:.3f}, {METRICS['auc_ci'][1]:.3f}]")

    # Compact grid + interpretations
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(ART / "roc.png"), use_column_width=True, caption="ROC curve (AUC)")
        st.markdown(
            f"- **AUC = {METRICS.get('auc', 0):.3f}** → good rank ordering.\n"
            f"- 95% CI **[{METRICS.get('auc_ci', [0,0])[0]:.3f}, {METRICS.get('auc_ci', [0,0])[1]:.3f}]** shows stability.\n"
            "Higher is better; curve above the diagonal indicates signal."
        )
    with c2:
        st.image(str(ART / "pr.png"), use_column_width=True, caption="Precision-Recall (AUPRC)")
        st.markdown(
            f"- **AUPRC = {METRICS.get('aupr', 0):.3f}** → strong precision at useful recalls.\n"
            "Use this when positives are rarer; watch precision drop as recall increases."
        )

    c3, c4 = st.columns(2)
    with c3:
        st.image(str(ART / "reliability.png"), use_column_width=True, caption="Reliability (Calibration)")
        st.markdown(
            f"- **Brier = {METRICS.get('brier', 0):.3f}** (lower is better).\n"
            "Blue dots near the orange diagonal indicate well-calibrated probabilities."
        )
    shap_img = ART / "shap_summary.png"
    with c4:
        if shap_img.exists():
            st.image(str(shap_img), use_column_width=True, caption="Global SHAP summary (top features)")
            st.markdown(
                "Features farther from zero have stronger global influence. "
                "Blue/pink scatter shows interactions across the dataset."
            )
        else:
            st.info("Global SHAP summary not bundled; local explanations still available on the **Explanations** page.")

    st.caption("We use **isotonic calibration** on a held-out fold to align predicted probabilities with observed frequencies.")
    license_footer()

# ---------- Page: BATCH SCORING ----------
elif page == "Batch Scoring":
    st.subheader("Score a CSV file")
    st.caption("Upload a CSV with the following columns (header required): " + ", ".join(FEATURES))
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            dfu = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in dfu.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                proba = MODEL.predict_proba(dfu[FEATURES])[:, 1]
                out = dfu.copy()
                out["risk_probability"] = proba
                lo_th = float(st.session_state.get("lo", 0.07))
                hi_th = float(st.session_state.get("hi", 0.35))
                out["risk_band"] = pd.cut(
                    out["risk_probability"],
                    bins=[-1, lo_th, hi_th, 1.1],
                    labels=["Low", "Medium", "High"]
                )
                st.dataframe(out.head(25), use_container_width=True)
                st.download_button(
                    "Download scored CSV",
                    out.to_csv(index=False).encode("utf-8"),
                    file_name="scored_heartrisk.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Could not score file: {e}")
    license_footer()

# ---------- Page: DATA EXPLORER ----------
elif page == "Data Explorer":
    st.subheader("Sample dataset (read-only)")
    if DF_BG is None:
        st.warning("Dataset file not found at data/heart.csv.")
    else:
        st.caption("These are the model input columns used for training & scoring.")
        st.dataframe(DF_BG.head(10), use_container_width=True)
        st.download_button(
            "Download sample CSV",
            DF_BG.to_csv(index=False).encode("utf-8"),
            file_name="heart.csv",
            mime="text/csv"
        )

        st.markdown("### Try a dataset row in TRIAGE")
        idx = st.number_input("Row index", min_value=0, max_value=len(DF_BG)-1, value=4, step=1)
        if st.button("Send selected row to TRIAGE"):
            push_row_to_triage(DF_BG.iloc[int(idx)][FEATURES].to_dict())
            st.rerun()

    license_footer()

# ---------- Page: PRIVACY ----------
else:
    st.subheader("Privacy-first & Trust")
    st.markdown("""
- No PHI used. Model trained on a public, de-identified dataset (UCI Heart).
- All predictions computed locally on this app/server.
- Transparent outputs: **calibrated probabilities** + **fairness dashboards**.
- **Limits**: small dataset; possible cohort bias; domain shift in other settings.
- This is **not** a diagnostic tool. Always consult a clinician.
    """)
    license_footer()