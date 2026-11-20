"""Shared pieces used by training, evaluation, prediction and the API."""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Feature sets. H3 compares DEMO with DEMO + BEHAVIOUR.
DEMO_NUM, DEMO_CAT = ["household_size"], ["age_group", "sex", "education", "livelihood", "residence", "province"]
BEH_NUM = ["loans_12m", "outstanding_loans", "repayment_burden", "debt_to_income",
           "uses_mobile_bank", "uses_mobile_money", "uses_digital_app", "uses_hustler_fund"]
BEH_CAT = ["purpose"]
CTRL_NUM, CTRL_CAT = ["shocks", "has_savings", "literacy_score", "nondigital_loans"], ["savings_difficulty"]

FEATURE_SETS = {
    "demo": (DEMO_NUM, DEMO_CAT),
    "demo_beh": (DEMO_NUM + BEH_NUM, DEMO_CAT + BEH_CAT),
    "full": (DEMO_NUM + BEH_NUM + CTRL_NUM, DEMO_CAT + BEH_CAT + CTRL_CAT),
}
SKEWED = ["loans_12m", "outstanding_loans", "repayment_burden", "debt_to_income", "household_size"]
LOG_COLS = ["loans_12m", "outstanding_loans"]


def columns(feature_set: str) -> list[str]:
    """All predictor columns (numeric then categorical) in a feature set."""
    num, cat = FEATURE_SETS[feature_set]
    return num + cat


class Winsorizer(TransformerMixin, BaseEstimator):
    """Caps each column at its 99th percentile, learned from the training data only."""

    def __init__(self, quantile=0.99):
        self.quantile = quantile

    def fit(self, X, y=None):
        self.caps_ = np.nanquantile(np.asarray(X, float), self.quantile, axis=0)
        return self

    def transform(self, X):
        return np.minimum(np.asarray(X, float), self.caps_)   # NaN stays NaN

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def log1p(x):
    """log(1 + x) with negative values clipped to 0, for skewed counts."""
    return np.log1p(np.clip(x, 0, None))


def make_preprocessor(feature_set: str, scale: bool) -> ColumnTransformer:
    """Cap + impute (with missing flags) + log for counts, scale for logistic regression, one-hot for categories."""
    num, cat = FEATURE_SETS[feature_set]
    scaler = [("scale", StandardScaler())] if scale else []

    def impute():
        return SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)

    logs = [c for c in num if c in LOG_COLS]
    capped = [c for c in num if c in SKEWED and c not in LOG_COLS]
    plain = [c for c in num if c not in SKEWED]
    parts = []
    if logs:
        parts.append(("counts", Pipeline([("cap", Winsorizer()), ("impute", impute()),
                                          ("log", FunctionTransformer(log1p, feature_names_out="one-to-one")),
                                          *scaler]), logs))
    if capped:
        parts.append(("skewed", Pipeline([("cap", Winsorizer()), ("impute", impute()), *scaler]), capped))
    if plain:
        parts.append(("numeric", Pipeline([("impute", impute()), *scaler]), plain))
    parts.append(("categorical", Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    return ColumnTransformer(parts)


# ---------------------------------------------------------------------------
# Using a saved model bundle: {"pipeline", "calibrator", "calibration", "threshold", "features"}
# ---------------------------------------------------------------------------
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibrate(calibrator, method: str, p_raw: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator ('platt' or 'isotonic') to raw model scores."""
    if method == "platt":
        return calibrator.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]
    return calibrator.predict(p_raw)


def predict_proba(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Calibrated probability of debt distress."""
    p_raw = bundle["pipeline"].predict_proba(X[bundle["features"]])[:, 1]
    return calibrate(bundle["calibrator"], bundle["calibration"], p_raw)


FEATURE_GROUPS = {
    "Borrowing frequency": ["loans_12m"],
    "Multiple borrowing": ["outstanding_loans", "uses_mobile_bank", "uses_mobile_money", "uses_digital_app",
                           "uses_hustler_fund"],
    "Repayment burden": ["repayment_burden", "debt_to_income"],
    "Borrowing purpose": ["purpose"],
    "Household shocks": ["shocks"], "Savings": ["has_savings", "savings_difficulty"],
    "Financial literacy": ["literacy_score"], "Non-digital loans": ["nondigital_loans"],
    "Age": ["age_group"], "Sex": ["sex"], "Education": ["education"], "Livelihood": ["livelihood"],
    "Household size": ["household_size"], "Location": ["residence", "province"],
}


def shap_by_group(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP values (log-odds) summed into the predictor groups above."""
    import shap

    pre, clf = pipeline.named_steps["pre"], pipeline.named_steps["clf"]
    Xt = pre.transform(X)
    names = pre.get_feature_names_out()
    if hasattr(clf, "coef_"):                                   # logistic regression: exact linear SHAP
        vals = (Xt - Xt.mean(axis=0)) * clf.coef_.ravel()
    else:
        with warnings.catch_warnings():                         # SHAP warns that LightGBM output is a list; handled below
            warnings.simplefilter("ignore", UserWarning)
            vals = shap.TreeExplainer(clf).shap_values(Xt)
        if isinstance(vals, list):
            vals = vals[1]
        elif np.ndim(vals) == 3:
            vals = vals[:, :, 1]
    vals = pd.DataFrame(vals, columns=names, index=X.index)
    out = {}
    for group, cols in FEATURE_GROUPS.items():
        # one-hot and missing-flag columns are named like "categorical__sex_Female"
        match = [n for n in names if any(n.split("__", 1)[1].replace("missingindicator_", "") == c
                                         or n.split("__", 1)[1].startswith(c + "_") for c in cols)]
        if match:
            out[group] = vals[match].sum(axis=1)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Turning an API request (plain fields) into model features
# ---------------------------------------------------------------------------
COUNTIES = ["Mombasa", "Kwale", "Kilifi", "Tana River", "Lamu", "Taita-Taveta", "Garissa", "Wajir", "Mandera",
            "Marsabit", "Isiolo", "Meru", "Tharaka-Nithi", "Embu", "Kitui", "Machakos", "Makueni", "Nyandarua",
            "Nyeri", "Kirinyaga", "Murang'a", "Kiambu", "Turkana", "West Pokot", "Samburu", "Trans-Nzoia",
            "Uasin Gishu", "Elgeyo-Marakwet", "Nandi", "Baringo", "Laikipia", "Nakuru", "Narok", "Kajiado", "Kericho",
            "Bomet", "Kakamega", "Vihiga", "Bungoma", "Busia", "Siaya", "Kisumu", "Homa Bay", "Migori", "Kisii",
            "Nyamira", "Nairobi City"]


def profile_to_features(profiles: list[dict]) -> pd.DataFrame:
    """Map API borrower profiles to the model's feature columns."""
    from Data.data_pipeline import province

    rows = []
    for p in profiles:
        income = p.get("monthly_income")
        age = p["age"]
        rows.append({
            "age_group": "16-24" if age <= 24 else "25-34" if age <= 34 else "35-44" if age <= 44
            else "45-54" if age <= 54 else "55+",
            "sex": p["sex"], "education": p["education"], "livelihood": p["livelihood"],
            "household_size": p["household_size"], "residence": p["residence"],
            "province": province(COUNTIES.index(p["county"]) + 1),
            "uses_mobile_bank": int(p["uses_mobile_bank"]), "uses_mobile_money": int(p["uses_mobile_money"]),
            "uses_digital_app": int(p["uses_digital_app"]), "uses_hustler_fund": int(p["uses_hustler_fund"]),
            "loans_12m": p.get("loans_past_12_months", np.nan),
            "outstanding_loans": p["outstanding_loans"],
            "repayment_burden": p["monthly_repayment"] / income if income and p.get("monthly_repayment") is not None
            else np.nan,
            "debt_to_income": (p.get("outstanding_balance") or 0) / income if income else np.nan,
            "purpose": p["main_purpose"], "shocks": p["shocks_past_year"],
            "savings_difficulty": p["savings_difficulty"], "has_savings": int(p["has_savings"]),
            "literacy_score": p["literacy_score"], "nondigital_loans": p["nondigital_loans"],
        })
    return pd.DataFrame(rows).astype({"loans_12m": float, "repayment_burden": float, "debt_to_income": float})
