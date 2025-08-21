# app.py — HeartRisk Assist (Medi-Hack 2025)
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

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
BUNDLE = load_bundle()                    # saved by train notebook/script
MODEL  = BUNDLE["model"]                  # CalibratedClassifierCV wrapping a Pipeline
FEATURES = BUNDLE["features"]
PREPROC  = BUNDLE.get("preprocessor", None)

METRICS = load_metrics()
DF_BG   = load_bg(FEATURES)

# ---------- Helpers ----------
SAMPLE = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0,
    "thalach": 150, "exang": 1, "oldpeak": 2.3, "slope": 0, "ca": 2, "thal": 3
}

def license_footer():
    st.markdown(
        "<hr/>"
        "<div style='font-size:12px'>MIT License — (c) 2025 Sweety Seelam. "
        "This tool is an educational triage prototype, not medical advice.</div>",
        unsafe_allow_html=True
    )

def get_underlying_estimator():
    # CalibratedClassifierCV(prefit) -> base_estimator is a Pipeline(pre + clf)
    return getattr(MODEL, "base_estimator", None) or MODEL

def get_pipeline_parts(est):
    # est may be a Pipeline with steps "pre" and "clf"
    pre = None; clf = None
    try:
        pre = est.named_steps.get("pre", None)
        clf = est.named_steps.get("clf", None)
    except Exception:
        pass
    return pre or PREPROC, clf or est

def risk_band(prob, lo, hi):
    if prob < lo: return "Low"
    if prob < hi: return "Medium"
    return "High"

def explain_single(x_df):
    """
    Returns (method_str, pd.DataFrame of top contributions) for a single-row input.
    If 'shap' available -> local SHAP. Else -> coef/importance-based fallback.
    """
    est = get_underlying_estimator()
    pre, clf = get_pipeline_parts(est)

    Xt = pre.transform(x_df[FEATURES])
    feat_names = pre.get_feature_names_out(FEATURES)

    # Dense row for fallbacks
    row_arr = Xt.toarray()[0] if hasattr(Xt, "toarray") else np.asarray(Xt)[0]

    # Try SHAP (optional)
    try:
        import shap
        # Background for linear explainer (small sample for speed)
        if DF_BG is not None:
            bg = DF_BG.sample(min(len(DF_BG), 200), random_state=42)
        else:
            bg = x_df[FEATURES].copy()
        Xbg = pre.transform(bg)
        if hasattr(Xbg, "toarray"): Xbg = Xbg.toarray()

        Xtd = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)

        if hasattr(clf, "estimators_"):             # RandomForest / tree model
            explainer = shap.TreeExplainer(clf)     # model_output='raw' (required)
            sv = explainer.shap_values(Xtd)         # list for binary: [neg, pos]
            vals = (np.asarray(sv[1])[0]
                    if isinstance(sv, list) else np.asarray(sv)[0])
            method = "SHAP (TreeExplainer)"
        else:                                       # LogisticRegression / linear
            explainer = shap.LinearExplainer(clf, Xbg)
            try:
                exp = explainer(Xtd)                # Explanation in newer versions
                vals = np.asarray(getattr(exp, "values", exp))[0]
            except Exception:
                vals = np.asarray(explainer.shap_values(Xtd))[0]
            method = "SHAP (LinearExplainer)"

        contrib = (pd.DataFrame({"feature": feat_names, "contribution": vals})
                     .sort_values("contribution", key=np.abs, ascending=False)
                     .head(15))
        return method, contrib

    except Exception:
        # Fallbacks if SHAP not installed or failed
        if hasattr(clf, "coef_"):
            vals = row_arr * clf.coef_[0]
            method = "Coefficient-based"
        elif hasattr(clf, "feature_importances_"):
            vals = row_arr * clf.feature_importances_
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
    st.rerun()

def push_row_to_triage(row_dict):
    for k, v in row_dict.items():
        st.session_state[k] = v
    # Jump to triage page
    st.session_state["nav"] = "Triage (Diagnostics)"
    st.rerun()

# ---------- Sidebar controls ----------
st.sidebar.header("Risk Thresholds")
lo = st.sidebar.slider("Low/Medium cutoff", 0.0, 0.5, 0.20, 0.01)
hi = st.sidebar.slider("Medium/High cutoff", 0.3, 0.9, 0.50, 0.01)
if hi <= lo:
    st.sidebar.error("Medium/High cutoff must be > Low/Medium cutoff.")
st.session_state["lo"] = lo
st.session_state["hi"] = hi

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "Triage (Diagnostics)", "Explanations", "Fairness (Ops)",
    "Model Quality", "Batch Scoring", "Data Explorer", "Privacy & Trust"
], key="nav")

# ---------- Page: TRIAGE ----------
if page == "Triage (Diagnostics)":
    st.subheader("Enter measurements")
    tips = {
        "sex": "0=female, 1=male",
        "cp": "Chest pain type (0–3)",
        "restecg": "ECG results (0–2)",
        "exang": "Exercise-induced angina (0/1)",
        "slope": "ST segment slope (0–2)",
        "ca": "# major vessels (0–3) colored by fluoroscopy",
        "thal": "0=unknown/NA, 1=fixed defect, 2=normal, 3=reversible defect"
    }

    with st.form("triage_form", clear_on_submit=False):
        cols = st.columns(3)
        vals = {}
        for i, f in enumerate(FEATURES):
            with cols[i % 3]:
                if f in ["sex","cp","fbs","restecg","exang","slope","ca","thal"]:
                    default = int(st.session_state.get(f, 0))
                    vals[f] = st.number_input(f, value=default, step=1, min_value=0, max_value=10,
                                              help=tips.get(f, ""), key=f)
                else:
                    default = float(st.session_state.get(f, 0.0))
                    vals[f] = st.number_input(f, value=default, step=0.1, format="%.2f", key=f)
        submitted = st.form_submit_button("Assess risk")

    if submitted:
        x = pd.DataFrame([vals])[FEATURES]
        prob = float(MODEL.predict_proba(x)[:, 1])   # pipeline + calibration inside
        band = risk_band(prob, lo, hi)
        st.metric("Calibrated risk", f"{prob:.1%}")
        st.success(f"Risk Band: **{band}** (cutoffs: <{lo:.0%}=Low, {lo:.0%}–{hi:.0%}=Medium, >{hi:.0%}=High)")
        st.caption("Note: Probabilities are **calibrated**. Explanations (next page) reflect the model before calibration.")

    st.button("Load sample (High risk)", type="secondary",
              help="Pre-fills fields with a typical higher-risk profile",
              on_click=fill_sample)
    license_footer()

# ---------- Page: EXPLANATIONS ----------
elif page == "Explanations":
    st.subheader("Why this decision?")
    st.caption("Local explanation for the most recent inputs. If none submitted yet, we use a representative sample.")

    # Build from current session values or fallback to SAMPLE
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
**Operational use:** monitor equity over time; if gaps are large, consider data collection, reweighting, or threshold tuning by cohort.
    """)
    license_footer()

# ---------- Page: MODEL QUALITY ----------
elif page == "Model Quality":
    st.subheader("Validation metrics")
    clean = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in METRICS.items()}
    st.write(clean)
    if "auc_ci" in METRICS:
        st.write(f"Test AUC 95% CI: [{METRICS['auc_ci'][0]:.3f}, {METRICS['auc_ci'][1]:.3f}]")
    st.image(str(ART / "roc.png"), caption="ROC curve (AUC)")
    st.image(str(ART / "pr.png"), caption="Precision-Recall (AUPRC)")
    st.image(str(ART / "reliability.png"), caption="Reliability (Calibration)")
    shap_img = ART / "shap_summary.png"
    if shap_img.exists():
        st.image(str(shap_img), caption="Global SHAP summary (top features)")
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
                lo_th = float(st.session_state.get("lo", 0.20))
                hi_th = float(st.session_state.get("hi", 0.50))
                out["risk_band"] = pd.cut(
                    out["risk_probability"],
                    bins=[-1, lo_th, hi_th, 1.1],
                    labels=["Low", "Medium", "High"]
                )
                st.dataframe(out.head(25))
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
        st.warning("`data/heart.csv` not found or unreadable in this deployment.")
        st.caption("Commit `data/heart.csv` to the repo (or provide another CSV with the same columns).")
    else:
        st.caption("These are the model input columns used for training & scoring.")
        st.dataframe(DF_BG.head(200))  # interactive grid (first 200 rows)

        # Download a small, shareable sample
        st.download_button(
            "Download sample CSV",
            DF_BG.head(1000).to_csv(index=False).encode("utf-8"),
            file_name="sample_heartrisk.csv",
            mime="text/csv"
        )

        # Let users pick a row and push it into the TRIAGE form
        st.markdown("#### Try a dataset row in TRIAGE")
        row_idx = st.number_input("Row index", min_value=0, max_value=len(DF_BG)-1, value=0, step=1)
        if st.button("Send selected row to TRIAGE"):
            row_vals = DF_BG.iloc[int(row_idx)][FEATURES].to_dict()
            push_row_to_triage(row_vals)
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