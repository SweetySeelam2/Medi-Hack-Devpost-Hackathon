# app.py — HeartRisk Assist (Medi-Hack 2025)
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

# ---------- Page setup ----------
st.set_page_config(page_title="HeartRisk Assist — Medi-Hack Edition", page_icon="❤️", layout="wide")
st.markdown("## HeartRisk Assist — Calibrated, Fair, Privacy-First Cardiac Risk Triage (Medi-Hack 2025)")
st.info("Built during Medi-Hack (Aug 18–23, 2025). Educational prototype — **not medical advice**. No PHI (Protected Health Information) used.")

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

# Use a de-duplicated view everywhere we show/select rows (model was trained on de-duped data ~302 rows)
if DF_BG is not None:
    DF_VIEW = DF_BG.drop_duplicates().reset_index(drop=True)
    RAW_N = len(DF_BG)
    DEDUP_N = len(DF_VIEW)
else:
    DF_VIEW, RAW_N, DEDUP_N = None, 0, 0

# ---------- Dictionaries & samples ----------
CATS = {
    "sex":    {0: "Female (0)", 1: "Male (1)"},
    "cp":     {0: "Typical angina (0)", 1: "Atypical angina (1)", 2: "Non-anginal pain (2)", 3: "Asymptomatic (3)"},
    "fbs":    {0: "Fasting blood sugar ≤ 120 mg/dL (0)", 1: "Fasting blood sugar > 120 mg/dL (1)"},
    "restecg":{0: "Normal (0)", 1: "ST-T wave abnormality (1)", 2: "Left ventricular hypertrophy (2)"},
    "exang":  {0: "No (0)", 1: "Yes (1)"},
    "slope":  {0: "Upsloping (0)", 1: "Flat (1)", 2: "Downsloping (2)"},
    "ca":     {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal":   {0: "Unknown (0)", 1: "Fixed defect (1)", 2: "Normal (2)", 3: "Reversible defect (3)"},
}
NUM_META = {
    "age":      {"label": "Age (years)",                    "min": 18, "max": 95,  "step": 1.0},
    "trestbps": {"label": "Resting blood pressure (mm Hg)", "min": 80, "max": 220, "step": 1.0},
    "chol":     {"label": "Serum cholesterol (mg/dL)",      "min": 100,"max": 600, "step": 1.0},
    "thalach":  {"label": "Maximum heart rate (bpm)",       "min": 60, "max": 230, "step": 1.0},
    "oldpeak":  {"label": "ST depression (mm)",             "min": 0.0,"max": 7.0, "step": 0.1},
}

# A base "strong risk" seed (we will auto-boost it to guarantee High if needed)
SAMPLE_SEED = {
    "age": 74, "sex": 1, "cp": 3, "trestbps": 170, "chol": 310, "fbs": 1, "restecg": 1,
    "thalach": 92, "exang": 1, "oldpeak": 5.0, "slope": 2, "ca": 3, "thal": 3
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
    if isinstance(model, CalibratedClassifierCV):
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

    pre = PREPROC
    if pre is None:
        raise RuntimeError("No preprocessor available for explanations.")
    if not hasattr(pre, "transformers_"):
        if DF_BG is None or DF_BG.empty:
            raise RuntimeError("Preprocessor is not fitted and no background data is available.")
        pre.fit(DF_BG[FEATURES])
    clf = est
    return pre, clf, None

def _predict_proba(clf, X):
    if hasattr(clf, "predict_proba"):
        return np.asarray(clf.predict_proba(X))[:, 1]
    if hasattr(clf, "decision_function"):
        z = np.asarray(clf.decision_function(X)).ravel()
        return 1 / (1 + np.exp(-z))
    raise RuntimeError("Classifier has neither predict_proba nor decision_function.")

def risk_band(prob, lo, hi):
    if prob < lo: return "Low"
    if prob < hi: return "Medium"
    return "High"

def signed_contrib_without_shap(pre, clf, x_df, feat_names):
    """
    Directional local contributions when SHAP isn't available.
    For each transformed feature j, replace it with the background mean and
    measure the probability drop: contrib[j] = base_prob - prob_with_feature_replaced.
    Positive => pushes risk UP; Negative => pushes risk DOWN.
    """
    Xt = pre.transform(x_df[FEATURES])
    Xrow = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    if DF_BG is not None:
        Xbg = pre.transform(DF_BG[FEATURES])
        Xbg = Xbg.toarray() if hasattr(Xbg, "toarray") else np.asarray(Xbg)
        mu = Xbg.mean(axis=0)
    else:
        mu = np.zeros(Xrow.shape[1])

    base = float(_predict_proba(clf, Xrow)[0])
    contribs = []
    for j in range(Xrow.shape[1]):
        Xalt = Xrow.copy()
        Xalt[0, j] = mu[j]
        p_alt = float(_predict_proba(clf, Xalt)[0])
        contribs.append(base - p_alt)

    df = (pd.DataFrame({"feature": feat_names, "contribution": contribs})
            .sort_values("contribution", key=np.abs, ascending=False).head(15))
    return "Directional (mean-replacement)", df

def explain_single(x_df):
    """
    Returns (method_str, pd.DataFrame of top contributions) for a single-row input.
    Tries SHAP first; falls back to directional contributions; then coefficients/importances.
    """
    pre, clf, _ = unwrap_to_fitted_pipeline(MODEL)
    Xt = pre.transform(x_df[FEATURES])
    feat_names = list(getattr(pre, "get_feature_names_out", lambda *_: FEATURES)(FEATURES))

    try:
        import shap
        bg = DF_VIEW.sample(min(len(DF_VIEW), 200), random_state=42) if DF_VIEW is not None else x_df[FEATURES].copy()
        Xbg = pre.transform(bg)
        if hasattr(Xbg, "toarray"): Xbg = Xbg.toarray()
        Xtd = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)

        if hasattr(clf, "estimators_"):
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(Xtd)
            vals = (np.asarray(sv[1])[0] if isinstance(sv, list) else np.asarray(sv)[0])
            method = "SHAP (TreeExplainer)"
        else:
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
        try:
            return signed_contrib_without_shap(pre, clf, x_df, feat_names)
        except Exception:
            Xtd = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
            row = Xtd[0]
            if hasattr(clf, "coef_"):
                vals = row * clf.coef_[0]
                method = "Coefficient-based"
            elif hasattr(clf, "feature_importances_"):
                vals = row * clf.feature_importances_
                method = "Importance-based (unsigned)"
            else:
                return "Unavailable", pd.DataFrame(columns=["feature", "contribution"])
            contrib = (pd.DataFrame({"feature": feat_names, "contribution": vals})
                       .sort_values("contribution", key=np.abs, ascending=False)
                       .head(15))
            return method, contrib

# --- High-risk sample utilities ---
def _predict_row(d):
    x = pd.DataFrame([d])[FEATURES]
    return float(MODEL.predict_proba(x)[:, 1])

def _clip_num(name, val):
    meta = NUM_META[name]
    return float(np.clip(val, meta["min"], meta["max"]))

def find_or_make_high_risk(hi_thresh=0.35):
    """
    Try: (1) seed sample, (2) boosted extremes grid, (3) highest-risk row in DF_VIEW.
    Return (row_dict, prob).
    """
    # 1) seed
    r = SAMPLE_SEED.copy()
    p = _predict_row(r)
    if p >= hi_thresh:
        return r, p

    # 2) small grid search over realistic extremes
    ages     = [70, 80, 90]
    chols    = [310, 400, 520]
    bps      = [170, 190, 210]
    thalachs = [60, 75, 90]
    oldpeaks = [4.0, 5.0, 6.5]
    best = (p, r.copy())
    for a in ages:
        for c in chols:
            for b in bps:
                for hr in thalachs:
                    for op in oldpeaks:
                        cand = {
                            "age": _clip_num("age", a), "sex": 1, "cp": 3, "trestbps": _clip_num("trestbps", b),
                            "chol": _clip_num("chol", c), "fbs": 1, "restecg": 2, "thalach": _clip_num("thalach", hr),
                            "exang": 1, "oldpeak": _clip_num("oldpeak", op), "slope": 2, "ca": 3, "thal": 3
                        }
                        pr = _predict_row(cand)
                        if pr > best[0]:
                            best = (pr, cand.copy())
                        if pr >= hi_thresh:
                            return cand, pr
    # 3) fallback: highest predicted in dataset view
    if DF_VIEW is not None and len(DF_VIEW) > 0:
        proba = MODEL.predict_proba(DF_VIEW[FEATURES])[:, 1]
        idx = int(np.argmax(proba))
        cand = DF_VIEW.iloc[idx][FEATURES].to_dict()
        pr = float(proba[idx])
        return cand, pr
    return r, p

def fill_sample(row_dict):
    """Fill the TRIAGE form with a given profile."""
    for k, v in row_dict.items():
        st.session_state[k] = int(v) if k in CATS else float(v)

def queue_nav(page_name: str):
    st.session_state["pending_nav"] = page_name

def apply_pending_nav():
    if "pending_nav" in st.session_state:
        target = st.session_state.pop("pending_nav")
        st.session_state["nav"] = target

def push_row_to_triage(row_dict):
    for k, v in row_dict.items():
        st.session_state[k] = v
    queue_nav("2) Triage (Diagnostics)")

# ---------- Sidebar (fixed policy + clear legends) ----------
st.sidebar.header("Risk Policy (fixed threshold ranges for demo)")
# Fixed operational thresholds for action bands (NOT changing the model)
DEFAULT_LO = 0.07   # Low if p < 7% (safe-to-monitor band when resources are tight)
DEFAULT_HI = 0.35   # High if p ≥ 35% (smaller subset for rapid work-up)

# store once for downstream pages
st.session_state["lo"] = float(DEFAULT_LO)
st.session_state["hi"] = float(DEFAULT_HI)

st.sidebar.markdown(
    f"""
**Band mapping (fixed):**  
- **Low:** p \< **{DEFAULT_LO:.0%}**  
- **Medium:** **{DEFAULT_LO:.0%}–{DEFAULT_HI:.0%}**  
- **High:** p ≥ **{DEFAULT_HI:.0%}**  

These **do not change the model** — they only map the calibrated probability to an operational action band used on **page 2 Triage** and **page 7 Batch Scoring**.  
**Why these values?** Demo defaults reflect a practical triage stance: keep **High** a smaller group to fast-track, and make **Low** conservative (<7%). In deployment, set bands from your own calibration, capacity, and miss-tolerance.
"""
)

with st.sidebar.expander("Triage measurements — code legend (for Page 2)", expanded=False):
    def _legend_line(name, mapping):
        items = "; ".join(str(v) for v in mapping.values())
        st.markdown(f"**{name}** — {items}")
    _legend_line("sex", CATS["sex"])
    _legend_line("cp (chest pain)", CATS["cp"])
    _legend_line("fbs (fasting blood sugar)", CATS["fbs"])
    _legend_line("restecg (rest ECG)", CATS["restecg"])
    _legend_line("exang (exercise-induced angina)", CATS["exang"])
    _legend_line("slope (ST segment slope)", CATS["slope"])
    _legend_line("ca (vessels by fluoroscopy)", CATS["ca"])
    _legend_line("thal (thalassemia)", CATS["thal"])

with st.sidebar.expander("What each measurement means & why included", expanded=False):
    st.markdown("""
- **age** — older age raises atherosclerotic risk.  
- **sex** — male sex shows higher CAD prevalence in classic cohorts.  
- **cp (chest pain type)** — typical/atypical angina signal strength; asymptomatic can still be risky.  
- **trestbps (rest BP)** — hypertension damages vessels → CAD risk.  
- **chol (serum cholesterol)** — higher LDL/total cholesterol → plaque formation.  
- **fbs (fasting blood sugar)** — proxy for diabetes (accelerates vascular disease).  
- **restecg** — resting ECG abnormalities suggest ischemia/hypertrophy.  
- **thalach (max HR)** — lower achieved HR can indicate limited reserve.  
- **exang (exercise-induced angina)** — provoked chest pain indicates supply–demand mismatch.  
- **oldpeak (ST depression)** — classic ischemia marker under stress.  
- **slope (ST slope)** — downsloping is more concerning.  
- **ca (vessels)** — greater number visualized ≈ more disease burden.  
- **thal** — perfusion category; reversible/fixed defects align with ischemia on imaging.
""")

NAV_OPTIONS = [
    "1) Project Overview",
    "2) Triage (Diagnostics)",
    "3) Explanations",
    "4) Fairness (Ops)",
    "5) Model Quality",
    "6) Data Explorer",
    "7) Batch Scoring",
    "8) Privacy & Trust",
    "9) Impact & Recommendations",
]

apply_pending_nav()
page = st.sidebar.radio("Go to", NAV_OPTIONS, key="nav")

# ---------- Page 1: PROJECT OVERVIEW ----------
if page == "1) Project Overview":
    st.subheader("Page 1 — Project Overview")

    st.markdown(f"""
### What this is
**HeartRisk Assist** estimates **calibrated probability** of cardiac disease from routine measurements (Kaggle **Heart Disease UCI** CSV).  
It’s for **triage**, not diagnosis — helping decide who needs **faster work-up**.

### Business problem
High volumes + scarce cardiology slots → over-triage wastes resources; under-triage is unsafe.  
We need **well-calibrated probabilities** (so *30% ≈ 30%*), **clear explanations**, and **fairness monitoring**.

### Goals
Calibrate probabilities; explain each prediction; monitor fairness; keep PHI out; use tunable (but fixed here) thresholds.

### Dataset
Loaded file rows: **{RAW_N}**. After removing duplicates used for modeling/analysis: **{DEDUP_N}**.  
Features: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`.  
Source: Kaggle (*Heart Disease Dataset*) — link on **Page 8**.

### How thresholds were chosen (7% / 35%)
These are **demo defaults**, not clinical truth. They encode a common ops policy:  
keep **High** a smaller subset to fast-track, and make **Low** truly low risk.

**Principled selection can be justified**
1) **Decide ops targets:** capacity for fast-track (e.g., 15–25% of cases), acceptable miss rate in **Low** (e.g., ≤3–5 per 100).  
2) **Use calibration to set two cutoffs:**  
   - **High:** choose the smallest *t* so only your top *C%* of cases have *p ≥ t*, or pick *t* so PPV among *p ≥ t* meets your target.  
   - **Low:** increase *t* from 0 until expected events among *p < t* stay within your allowed misses per 100.  
   On calibrated models this often lands **Low** around **0.05–0.10** and **High** around **0.30–0.40** — hence **0.07 / 0.35** as starters.  
3) **Optional single-cutoff (2-way):** *t* = *C*<sub>FP</sub> / (*C*<sub>FP</sub> + *C*<sub>FN</sub>).  
   If a miss is ~10× worse than an unnecessary test, *t* ≈ **0.09**.

Default value ranges are **7% / 35%** in this demo. Replace them with site-specific thresholds derived from **your** validation data, capacity, and risk tolerance.
""")
    st.markdown("""
### Glossary
- **AUC** = Area Under ROC Curve; **AUPRC** = Area Under Precision–Recall Curve  
- **TPR** = True Positive Rate (sensitivity); **FPR** = False Positive Rate (fall-out)  
- **PHI** = Protected Health Information
""")
    license_footer()

# ---------- Page 2: TRIAGE ----------
elif page == "2) Triage (Diagnostics)":
    st.subheader("Page 2 — Triage (Diagnostics)")

    with st.expander("How to use", expanded=False):
        st.markdown(f"""
1) Enter the measurements (or click **Load high-risk sample**).  
2) Press **Assess risk** — you’ll get a **calibrated probability** and an **action band** based on the fixed policy (**Low < {DEFAULT_LO:.0%}**, **Medium {DEFAULT_LO:.0%}–{DEFAULT_HI:.0%}**, **High ≥ {DEFAULT_HI:.0%}**).  
3) See **page 3 Explanations** for what pushed risk **up** or **down**.
        """)

    tips = {
        "sex": "Biological sex (0 = Female, 1 = Male). Epidemiology differs by sex → affects baseline risk.",
        "cp": "Chest pain type at presentation; typical angina weighs stronger for CAD risk.",
        "restecg": "Resting electrocardiogram result; abnormalities suggest ischemia/hypertrophy.",
        "exang": "Exercise-induced angina during test (0=No, 1=Yes).",
        "slope": "ST segment slope at peak exercise; downsloping is more concerning.",
        "ca": "Number of major vessels (0–3) colored by fluoroscopy; higher counts ≈ greater disease burden.",
        "thal": "Perfusion/thalassemia category; abnormal defects align with ischemia on imaging.",
        "thalach": "Max heart rate achieved (bpm); lower values can indicate limited reserve.",
        "oldpeak": "ST depression (mm) relative to rest; larger depression indicates ischemia.",
        "trestbps": "Systolic blood pressure at rest (mm Hg); hypertension increases atherosclerotic risk.",
        "chol": "Serum cholesterol (mg/dL); higher levels increase plaque formation.",
        "age": "Age in years; risk increases with age.",
    }

    def option_caption(key):
        if key not in CATS: return ""
        return "Options: " + "; ".join(str(v) for v in CATS[key].values())

    with st.form("triage_form", clear_on_submit=False):
        cols = st.columns(3)
        vals = {}
        for i, f in enumerate(FEATURES):
            with cols[i % 3]:
                if f in CATS:
                    default_code = int(st.session_state.get(f, SAMPLE_SEED.get(f, list(CATS[f].keys())[0])))
                    vals[f] = st.selectbox(
                        f, options=list(CATS[f].keys()),
                        index=list(CATS[f].keys()).index(default_code),
                        format_func=lambda k, _f=f: CATS[_f][k],
                        help=tips.get(f, "")
                    )
                    st.caption(tips.get(f, ""))
                    st.caption(option_caption(f))
                else:
                    meta = NUM_META.get(f, {"label": f, "min": 0.0, "max": 9999.0, "step": 1.0})
                    default_val = float(st.session_state.get(f, SAMPLE_SEED.get(f, 0.0)))
                    vals[f] = st.number_input(
                        meta["label"] if meta["label"] else f,
                        value=default_val,
                        min_value=float(meta["min"]),
                        max_value=float(meta["max"]),
                        step=float(meta["step"]),
                        help=tips.get(f, "")
                    )
                    st.caption(tips.get(f, ""))

        submitted = st.form_submit_button("Assess risk")

    if submitted:
        x = pd.DataFrame([vals])[FEATURES]
        prob = float(MODEL.predict_proba(x)[:, 1])
        st.session_state["last_inputs"] = vals
        st.session_state["last_prob"] = prob

    if "last_prob" in st.session_state:
        lo, hi = DEFAULT_LO, DEFAULT_HI
        prob = float(st.session_state["last_prob"])
        band = risk_band(prob, lo, hi)

        per100 = int(round(prob * 100))
        if prob < lo:
            relation, cut = "<", f"{lo:.0%}"
            meaning = "Very low predicted probability — typically safe to monitor with routine follow-up."
        elif prob < hi:
            relation, cut = "between", f"{lo:.0%} and {hi:.0%}"
            meaning = "Intermediate probability — consider clinician review, additional tests, or short-interval follow-up."
        else:
            relation, cut = "≥", f"{hi:.0%}"
            meaning = "Elevated probability — prioritize faster work-up or referral."

        st.metric("Calibrated risk", f"{prob:.1%}")
        st.success(f"**Risk Band: {band}** — because **{prob:.1%}** {relation} {cut}.")
        st.markdown(
            f"- **Interpretation:** Among 100 similar patients, about **{per100}** would be expected to have cardiac disease.\n"
            f"- **Band meaning:** {meaning}\n"
            f"- **Next:** open **page 3 Explanations** to see which features pushed the risk up or down for this case."
        )

    if st.button("Load high-risk sample", type="secondary",
                 help="Auto-loads a profile that scores in **High** (≥ 35%) when possible for this trained model."):
        row, pr = find_or_make_high_risk(DEFAULT_HI)
        fill_sample(row)
        st.session_state["last_inputs"] = row
        st.session_state["last_prob"] = pr
        st.rerun()

    license_footer()

# ---------- Page 3: EXPLANATIONS ----------
elif page == "3) Explanations":
    st.subheader("Page 3 — Why this decision?")
    st.caption("Local explanation for the most recent inputs. If none submitted yet, we use a representative sample.")
    row = {f: st.session_state.get(f, SAMPLE_SEED.get(f, 0)) for f in FEATURES}
    x = pd.DataFrame([row])[FEATURES]
    method, contrib = explain_single(x)

    if "last_prob" in st.session_state:
        st.info(f"Last predicted **calibrated risk**: **{st.session_state['last_prob']:.1%}**")

    if contrib.empty:
        st.warning("Explanation currently unavailable for this configuration.")
    else:
        st.write(f"Explanation method: **{method}**")

        # Vertical signed bar chart (up = higher risk, down = lower); labels straight
        chart = (
            alt.Chart(contrib)
              .mark_bar()
              .encode(
                  x=alt.X("feature:N", sort=None, axis=alt.Axis(title=None, labelAngle=0)),
                  y=alt.Y("contribution:Q", axis=alt.Axis(title="Contribution to risk (signed)")),
                  tooltip=["feature", alt.Tooltip("contribution:Q", format=".4f")]
              )
        )
        zero_line = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(strokeDash=[4,4]).encode(y="y:Q")
        st.altair_chart((zero_line + chart).properties(height=420), use_container_width=True)

        st.markdown("#### Top drivers for this case")
        for _, r in contrib.head(5).iterrows():
            direction = "↑ increases" if r["contribution"] > 0 else "↓ decreases"
            st.markdown(f"- **{r['feature']}**: {direction} risk by ~{abs(r['contribution']):.3f} (model-space units).")

        st.caption(
            "Bars above zero pushed the probability upward; bars below zero pulled it down. "
            "Use this to justify next steps (e.g., large **oldpeak** or **downsloping slope** supporting rapid work-up)."
        )
    license_footer()

# ---------- Page 4: FAIRNESS ----------
elif page == "4) Fairness (Ops)":
    st.subheader("Page 4 — Slice metrics (AUC)")
    st.caption("We report AUC by **sex**, **age buckets** (<45, 45–60, >60), and **cp** (chest pain type).")

    auc_by_sex = METRICS.get("auc_by_sex", {})
    auc_by_age = METRICS.get("auc_by_age_bucket", {})
    auc_by_cp  = METRICS.get("auc_by_cp", {})

    st.json({"auc_by_sex": auc_by_sex, "auc_by_age_bucket": auc_by_age, "auc_by_cp": auc_by_cp})

    def getv(d, k):
        try: return float(d.get(k, np.nan))
        except: return np.nan

    fem = getv(auc_by_sex, "0")
    male = getv(auc_by_sex, "1")
    a_45_60 = getv(auc_by_age, "45to60")
    a_gt60  = getv(auc_by_age, "gt60")
    cp0     = getv(auc_by_cp, "0")

    st.markdown("#### What this means")
    st.markdown(
        f"- **Sex**: Female **AUC {fem:.3f}**, Male **AUC {male:.3f}** → gap **{abs(male - fem):.3f}**.\n"
        f"- **Age**: 45–60 **AUC {a_45_60:.3f}** vs >60 **AUC {a_gt60:.3f}** → older cohort shows **lower discrimination**.\n"
        f"- **Chest pain 0 (Typical angina)** slice **AUC {cp0:.3f}** (others similar in JSON)."
    )
    st.markdown("""
**Operational guidance**
- Track slice AUCs over time; investigate gaps ≳0.05–0.10.
- Mitigations: collect more data for under-served cohorts, reweight during training,
  or use **cohort-specific thresholds** with clinical oversight.
- Post-deployment: audit periodically for drift and equity.
    """)
    license_footer()

# ---------- Page 5: MODEL QUALITY ----------
elif page == "5) Model Quality":
    st.subheader("Page 5 — Validation metrics")
    clean = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in METRICS.items()}
    st.write(clean)
    if "auc_ci" in METRICS:
        st.write(f"Test AUC 95% CI: [{METRICS['auc_ci'][0]:.3f}, {METRICS['auc_ci'][1]:.3f}]")

    c1, c2 = st.columns(2)
    with c1:
        st.image(str(ART / "roc.png"), use_column_width=True, caption="ROC curve (AUC)")
        st.markdown(
            f"- **AUC = {METRICS.get('auc', 0):.3f}** → ranks a random positive above a random negative about **{METRICS.get('auc', 0):.0%}** of the time.\n"
            "- **How to use:** choose a probability threshold balancing **TPR** (*True Positive Rate*, sensitivity) vs **FPR** (*False Positive Rate*, fall-out). "
            "For screening, accept higher FPR to keep TPR high; for scarce resources, move the threshold right."
        )
    with c2:
        st.image(str(ART / "pr.png"), use_column_width=True, caption="Precision–Recall (AUPRC)")
        st.markdown(
            f"- **AUPRC = {METRICS.get('aupr', 0):.3f}** → average precision across recalls; baseline equals class prevalence.\n"
            "- **How to use:** when positives are rarer, PR is more informative than ROC; ensure precision remains acceptable "
            "at the recall level your clinic needs."
        )

    c3, c4 = st.columns(2)
    with c3:
        st.image(str(ART / "reliability.png"), use_column_width=True, caption="Reliability (Calibration)")
        st.markdown(
            f"- **Brier score = {METRICS.get('brier', 0):.3f}** (lower is better). "
            "Dots near the diagonal mean **0.30 ≈ 30%** observed rate, enabling meaningful **Low/Med/High bands**."
        )
    shap_img = ART / "shap_summary.png"
    with c4:
        if shap_img.exists():
            st.image(str(shap_img), use_column_width=True, caption="Global SHAP summary (top features)")
            st.markdown(
                "Bigger magnitude → stronger global influence; color scatter shows interactions. "
                "Sanity-check that medically plausible drivers (e.g., **oldpeak**, **slope**, **thal**) matter."
            )
        else:
            st.info("Global SHAP summary not bundled; local explanations are on **3) Explanations**.")

    st.caption("We use **isotonic calibration** on a held-out fold to align predicted probabilities with observed frequencies.")
    license_footer()

# ---------- Page 6: DATA EXPLORER ----------
elif page == "6) Data Explorer":
    st.subheader("Page 6 — Sample dataset (read-only)")
    if DF_VIEW is None or len(DF_VIEW) == 0:
        st.warning("Dataset file not found at data/heart.csv.")
    else:
        n = len(DF_VIEW)
        st.caption("These are the model input columns used for training & scoring (de-duplicated; no PHI).")
        st.dataframe(DF_VIEW.head(200), use_container_width=True)
        st.download_button(
            "Download sample CSV",
            DF_VIEW.to_csv(index=False).encode("utf-8"),
            file_name="heart_dedup.csv",
            mime="text/csv"
        )

        st.markdown("### Try a dataset row in TRIAGE")
        st.caption(
            f"Select a row index **(0–{n-1})** from the de-duplicated training CSV and send it to **2) Triage**. "
            "Each row represents a single **patient encounter** with the input measurements above. "
            "Sending it to TRIAGE lets you see how a real profile scores and which features drive that decision."
        )
        idx = st.selectbox(f"Row index (0–{n-1}):", options=list(range(n)), index=min(4, n-1))
        if st.button("Send selected row to TRIAGE"):
            push_row_to_triage(DF_VIEW.iloc[int(idx)][FEATURES].to_dict())
            st.rerun()

    license_footer()

# ---------- Page 7: BATCH SCORING ----------
elif page == "7) Batch Scoring":
    st.subheader("Page 7 — Score a CSV file")
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
                lo_th = DEFAULT_LO
                hi_th = DEFAULT_HI
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
                st.caption(f"`risk_band` uses the fixed policy: Low < {lo_th:.0%}, Medium {lo_th:.0%}–{hi_th:.0%}, High ≥ {hi_th:.0%}.")
        except Exception as e:
            st.error(f"Could not score file: {e}")
    license_footer()

# ---------- Page 8: PRIVACY ----------
elif page == "8) Privacy & Trust":
    st.subheader("Page 8 — Privacy & Trust (why this matters)")
    st.markdown("""
**What we do**
- Train on the public, de-identified **Kaggle Heart Disease Dataset** (no PHI).
- Compute predictions locally in this app/server; no patient data leaves your session.
- Expose **calibrated probabilities**, **fairness slices**, and **explanations** for transparency.

**What we don't do**
- No storage of PHI (Protected Health Information) or any uploaded files beyond this session.
- No black-box scores without explanation; no automated clinical decisions.

**Limits & responsible use**
- Small dataset → potential cohort bias; domain shift likely at new sites.
- Use with clinical oversight; log decisions; audit for drift and equity.

**Dataset link (as used here):**  
[Kaggle — Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
""")
    license_footer()

# ---------- Page 9: IMPACT ----------
else:
    st.subheader("Page 9 — Impact & Recommendations")
    st.markdown("""
### Summary & Conclusions
- **Model quality:** **AUC = {auc:.3f}**, **AUPRC = {aupr:.3f}**, **Brier = {brier:.3f}**, **AUC 95% CI = [{lo_ci:.3f}, {hi_ci:.3f}]**.  
- **Calibration:** good after isotonic fitting → probabilities are usable for operational bands.  
- **Equity watchpoint:** older patients (>60) show lower discrimination (**AUC {age_gt60:.3f}**) vs **45–60** (**AUC {age_45_60:.3f}**).

### What we actually solved
- **Business problem:** crowded clinics need to **prioritize** who gets faster work-up.  
- **Our approach:** provide a **calibrated probability** (not just a score), a **clear banding policy**, and **per-case explanations** so staff understand the drivers.  
- **Outcome:** staff can route **High** cases sooner, safely **monitor Low**, and investigate **Medium** with context — reducing unnecessary follow-ups while keeping sensitivity.

### Operational Value (order-of-magnitude)
- Assume **2,000 visits/month** and **15%** suspected cardiac → **300** triage cases.  
- Average downstream test cost **$600** each.  
- If calibrated triage reduces unnecessary follow-ups by **5–8%** (holding sensitivity), that’s **15–24** fewer tests → **$9k–$14.4k/month** saved, plus clinician time.

### Recommendations
1. **Adopt fixed, calibrated thresholds** tuned with clinicians (demo uses **7%/35%**).  
2. **Monitor fairness** — track slice AUCs and consider **cohort-specific thresholds** if gaps persist.  
3. **Governance** — IRB review, decision logs, drift checks, periodic re-training on local EHR cohorts.  
4. **Next data step** — retrain with site data, add richer features (labs, meds), and run prospective A/B tests.

### Who benefits
- Health systems (e.g., **Kaiser Permanente, HCA, Cleveland Clinic, Mayo Clinic**).  
- Payer-provider orgs (**UnitedHealth/Optum, Humana, CVS/Aetna**).  
- Digital triage / telehealth vendors needing transparent, calibrated risk support.

*Educational prototype — not medical advice.*
    """.format(
        auc=METRICS.get("auc", float("nan")),
        aupr=METRICS.get("aupr", float("nan")),
        brier=METRICS.get("brier", float("nan")),
        lo_ci=METRICS.get("auc_ci", [float("nan"), float("nan")])[0],
        hi_ci=METRICS.get("auc_ci", [float("nan"), float("nan")])[1],
        age_gt60=float(METRICS.get("auc_by_age_bucket", {}).get("gt60", float("nan"))),
        age_45_60=float(METRICS.get("auc_by_age_bucket", {}).get("45to60", float("nan")))
    ))
    license_footer()