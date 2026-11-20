"""Model training: tuning, cross-validated comparison, selection, calibration and the final model.

All choices are made on the training clusters only; the test set is used in evaluate.py.
Run from the project root (after Data/data_pipeline.py):
    python -m Model.train
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from Model.components import _logit, calibrate, columns, make_preprocessor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "processed" / "model_data.parquet"
MODEL_DIR = ROOT / "Model"
RESULTS = MODEL_DIR / "results"
SEED = 2026
N_ITER = 30            # random-search configurations per algorithm
N_FOLDS, N_REPEATS = 5, 3
MIN_LEAF = 30          # every tree leaf must describe at least 30 borrowers
NAMES = {"dummy": "Baseline (prevalence)", "logistic": "Logistic regression", "random_forest": "Random forest",
         "xgboost": "XGBoost", "lightgbm": "LightGBM"}
SIMPLICITY = ["logistic", "random_forest", "lightgbm", "xgboost"]


def load_train() -> pd.DataFrame:
    """Load the modelling table and keep the training rows only."""
    df = pd.read_parquet(DATA)
    return df[~df["is_test"]].reset_index(drop=True)


def make_folds(df: pd.DataFrame, repeats: int = N_REPEATS) -> list:
    """Stratified group K-fold: whole clusters stay in one fold, distress rate similar across folds."""
    folds = []
    for r in range(repeats):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + r)
        folds += [(r, tr, va) for tr, va in cv.split(df, df["distress"], df["cluster"])]
    return folds


def sample_params(algo: str, rng) -> dict:
    """One random configuration from each algorithm's search space."""
    if algo == "logistic":
        return {"C": float(10 ** rng.uniform(-3, 1)), "penalty": str(rng.choice(["l1", "l2"]))}
    if algo == "random_forest":
        return {"n_estimators": int(rng.choice([300, 500])), "max_features": str(rng.choice(["sqrt", "log2"])),
                "max_depth": None if rng.random() < 0.5 else int(rng.integers(4, 16)),
                "min_samples_leaf": int(rng.integers(MIN_LEAF, 80))}
    if algo == "xgboost":
        return {"n_estimators": int(rng.integers(100, 600)), "learning_rate": float(10 ** rng.uniform(-2, -0.7)),
                "max_depth": int(rng.integers(2, 7)), "subsample": float(rng.uniform(0.6, 1)),
                "colsample_bytree": float(rng.uniform(0.5, 1)), "reg_lambda": float(10 ** rng.uniform(-1, 1.5)),
                "min_child_weight": float(rng.uniform(7, 30))}
    if algo == "lightgbm":
        return {"n_estimators": int(rng.integers(100, 600)), "learning_rate": float(10 ** rng.uniform(-2, -0.7)),
                "num_leaves": int(rng.integers(4, 48)), "subsample": float(rng.uniform(0.6, 1)),
                "colsample_bytree": float(rng.uniform(0.5, 1)), "reg_lambda": float(10 ** rng.uniform(-1, 1.5)),
                "min_child_samples": int(rng.integers(MIN_LEAF, 100))}
    return {}


def make_model(algo: str, feature_set: str, params: dict) -> Pipeline:
    """Preprocessing + classifier in one Pipeline, so preprocessing is refitted inside every fold."""
    if algo == "dummy":
        clf = DummyClassifier(strategy="prior")
    elif algo == "logistic":
        clf = LogisticRegression(C=params["C"], solver="liblinear" if params["penalty"] == "l1" else "lbfgs",
                                 l1_ratio=1.0 if params["penalty"] == "l1" else 0.0, max_iter=2000)
    elif algo == "random_forest":
        clf = RandomForestClassifier(**params, random_state=SEED, n_jobs=-1)
    elif algo == "xgboost":
        clf = XGBClassifier(**params, tree_method="hist", random_state=SEED, n_jobs=-1)
    else:
        clf = LGBMClassifier(**params, subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1)
    return Pipeline([("pre", make_preprocessor(feature_set, scale=(algo == "logistic"))), ("clf", clf)])


def weighted_scores(y, p, w) -> dict:
    """Survey-weighted metrics (unweighted metrics would describe the sample, not the population)."""
    return {"pr_auc": average_precision_score(y, p, sample_weight=w),
            "roc_auc": roc_auc_score(y, p, sample_weight=w),
            "brier": brier_score_loss(y, p, sample_weight=w)}


def cross_validate(algo, feature_set, params, df, folds):
    """Out-of-fold predictions (one row per repeat) and a score per fold."""
    X, y, w = df[columns(feature_set)], df["distress"].to_numpy(), df["weight"].to_numpy()
    oof = np.full((max(f[0] for f in folds) + 1, len(df)), np.nan)
    scores = []
    for i, (r, tr, va) in enumerate(folds):
        model = make_model(algo, feature_set, params).fit(X.iloc[tr], y[tr])
        p = model.predict_proba(X.iloc[va])[:, 1]
        oof[r, va] = p
        scores.append({"repeat": r, "fold": i % N_FOLDS, **weighted_scores(y[va], p, w[va])})
    return oof, pd.DataFrame(scores)


def tune(algo, feature_set, df, folds, n_iter=N_ITER):
    """Random search scored by mean weighted PR-AUC on the folds of the first repeat."""
    rng = np.random.default_rng(SEED)
    tuning_folds = [f for f in folds if f[0] == 0]
    trials = []
    for i in range(n_iter):
        params = sample_params(algo, rng)
        _, s = cross_validate(algo, feature_set, params, df, tuning_folds)
        trials.append({"trial": i, "pr_auc": s["pr_auc"].mean(), "params": params})
    trials = pd.DataFrame(trials)
    return trials.loc[trials["pr_auc"].idxmax(), "params"], trials


def select_model(summary: pd.DataFrame, margin=0.01) -> str:
    """Best mean PR-AUC, but prefer a simpler model within 0.01 of the best."""
    best = summary["pr_auc_mean"].max()
    close = summary[summary["pr_auc_mean"] >= best - margin].index
    return min(close, key=SIMPLICITY.index)


def cluster_bootstrap(df, n_boot=500, seed=SEED):
    """Row indices for resamples that draw whole clusters (within county)."""
    rng = np.random.default_rng(seed)
    rows = df.groupby("cluster").indices
    by_county = df.groupby("county")["cluster"].unique()
    for _ in range(n_boot):
        yield np.concatenate([rows[c] for cl in by_county for c in rng.choice(cl, len(cl))])


def h3_gain(df, oof_demo, oof_beh, n_boot=500):
    """PR-AUC gain from adding behaviour to demographics, with a 95% cluster-bootstrap interval."""
    y, w = df["distress"].to_numpy(), df["weight"].to_numpy()

    def gain(idx):
        return np.mean([average_precision_score(y[idx], oof_beh[r, idx], sample_weight=w[idx])
                        - average_precision_score(y[idx], oof_demo[r, idx], sample_weight=w[idx])
                        for r in range(oof_demo.shape[0])])

    boots = [gain(idx) for idx in cluster_bootstrap(df, n_boot)]
    return {"gain": gain(np.arange(len(df))), "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5)}


def fit_calibrator(p, y, w, groups):
    """Compare Platt scaling and isotonic regression on a cluster split; keep the lower Brier score."""
    rng = np.random.default_rng(SEED)
    held = rng.choice(np.unique(groups), size=len(np.unique(groups)) // 2, replace=False)
    ev = np.isin(groups, held)
    fits = {"platt": lambda m: LogisticRegression(C=1e6).fit(_logit(p[m]).reshape(-1, 1), y[m], sample_weight=w[m]),
            "isotonic": lambda m: IsotonicRegression(out_of_bounds="clip").fit(p[m], y[m], sample_weight=w[m])}
    brier = {k: brier_score_loss(y[ev], calibrate(f(~ev), k, p[ev]), sample_weight=w[ev]) for k, f in fits.items()}
    brier["uncalibrated"] = brier_score_loss(y[ev], p[ev], sample_weight=w[ev])
    method = min(("platt", "isotonic"), key=brier.get)
    return fits[method](np.ones(len(p), bool)), method, brier


def best_threshold(y, p, w):
    """Threshold that maximises weighted F1."""
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    return float(grid[np.argmax([f1_score(y, p >= t, sample_weight=w) for t in grid])])


def summarise(folds_df: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of the fold scores for each algorithm and feature set, best first."""
    return folds_df.groupby(["algorithm", "feature_set"]).agg(
        pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"), roc_auc_mean=("roc_auc", "mean"),
        brier_mean=("brier", "mean")).reset_index().sort_values("pr_auc_mean", ascending=False)


def main():
    """Tune, compare and select the models, then calibrate and save the chosen one."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_train()
    folds = make_folds(df)
    y, w, g = df["distress"].to_numpy(), df["weight"].to_numpy(), df["cluster"].to_numpy()

    # 1. Tune each algorithm on the full feature set, and the demographic baseline.
    best, oof, fold_scores = {}, {}, []
    for algo in SIMPLICITY:
        best[("full", algo)], trials = tune(algo, "full", df, folds)
        trials.to_csv(RESULTS / f"tuning_{algo}.csv", index=False)
        print(f"Tuned {NAMES[algo]}: best tuning PR-AUC {trials['pr_auc'].max():.3f}")
    best[("demo", "logistic")], _ = tune("logistic", "demo", df, folds)
    best[("demo", "dummy")] = {}

    # 2. Compare on all folds.
    for (fs, algo), params in best.items():
        oof[(fs, algo)], s = cross_validate(algo, fs, params, df, folds)
        fold_scores.append(s.assign(model=NAMES[algo], algorithm=algo, feature_set=fs))
    folds_df = pd.concat(fold_scores)
    summary = summarise(folds_df)

    # 3. Select.
    full = summary[summary["feature_set"] == "full"].set_index("algorithm")
    chosen = select_model(full)
    print(f"Chosen model: {NAMES[chosen]}")

    # 4. H3: demographics vs demographics + behaviour for the chosen algorithm.
    for fs in ("demo", "demo_beh"):
        if (fs, chosen) not in best:
            best[(fs, chosen)], _ = tune(chosen, fs, df, folds)
            oof[(fs, chosen)], s = cross_validate(chosen, fs, best[(fs, chosen)], df, folds)
            folds_df = pd.concat([folds_df, s.assign(model=NAMES[chosen], algorithm=chosen, feature_set=fs)])
    h3 = h3_gain(df, oof[("demo", chosen)], oof[("demo_beh", chosen)])
    print(f"H3 gain in PR-AUC: {h3['gain']:.3f} (95% CI {h3['ci_low']:.3f} to {h3['ci_high']:.3f})")

    # 5. Calibrate and choose the threshold on stacked out-of-fold predictions.
    p = oof[("full", chosen)].reshape(-1)
    yy, ww, gg = np.tile(y, N_REPEATS), np.tile(w, N_REPEATS), np.tile(g, N_REPEATS)
    calibrator, method, brier = fit_calibrator(p, yy, ww, gg)
    threshold = best_threshold(yy, calibrate(calibrator, method, p), ww)
    print(f"Calibration: {method}; threshold {threshold:.3f}")

    # 6. Refit every model on the full training set and save.
    finals = {(fs, a): make_model(a, fs, prm).fit(df[columns(fs)], y) for (fs, a), prm in best.items()}
    bundle = {"pipeline": finals[("full", chosen)], "calibrator": calibrator, "calibration": method,
              "threshold": threshold, "features": columns("full"), "algorithm": chosen}
    joblib.dump(bundle, MODEL_DIR / "model.joblib", compress=3)
    joblib.dump(finals, RESULTS / "all_models.joblib", compress=3)
    summarise(folds_df).to_csv(RESULTS / "cv_summary.csv", index=False)   # includes the H3 runs from step 4
    folds_df.to_csv(RESULTS / "cv_folds.csv", index=False)
    np.save(RESULTS / "oof_chosen.npy", oof[("full", chosen)])
    meta = {"model": NAMES[chosen], "params": best[("full", chosen)], "calibration": method,
            "calibration_brier": brier, "threshold": threshold, "features": columns("full"), "h3": h3,
            "label": "Late/missed/no repayment (E2E 1-2) AND a hardship sacrifice (E2D 1, 3, 4, 7), past 12 months",
            "population": "Current digital credit borrowers, FinAccess 2024",
            "intended_use": "Group-level risk profiles; not for decisions about individual borrowers"}
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=1, default=float))
    print("Saved Model/model.joblib and Model/metadata.json")


if __name__ == "__main__":
    main()
