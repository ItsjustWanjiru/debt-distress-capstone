"""Evaluation on the held-out test clusters (used once, after training is finished).

Produces test metrics with 95% cluster-bootstrap intervals, a comparison of all models,
curves, a confusion matrix, SHAP importance, a fairness check and a county hold-out test.
Run from the project root (after Model/train.py):
    python -m Model.evaluate
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold

from Model.components import columns, predict_proba, shap_by_group
from Model.train import DATA, MODEL_DIR, NAMES, RESULTS, SEED, cluster_bootstrap, make_model, weighted_scores

BLUE, ORANGE, GREY = "#1F4E79", "#C55A11", "#7F7F7F"


def threshold_scores(y, p, w, t) -> dict:
    """Weighted precision, recall, F1 and share flagged at threshold t."""
    pred = p >= t
    return {"precision": precision_score(y, pred, sample_weight=w), "recall": recall_score(y, pred, sample_weight=w),
            "f1": f1_score(y, pred, sample_weight=w), "share_flagged": np.average(pred, weights=w)}


def test_metrics(test, p_raw, p_cal, threshold, n_boot=1000) -> pd.DataFrame:
    """Point estimates and 95% cluster-bootstrap intervals on the test set."""
    y, w = test["distress"].to_numpy(), test["weight"].to_numpy()

    def all_scores(idx):
        s = weighted_scores(y[idx], p_raw[idx], w[idx])            # ranking on raw scores
        s["brier"] = weighted_scores(y[idx], p_cal[idx], w[idx])["brier"]
        return s | threshold_scores(y[idx], p_cal[idx], w[idx], threshold)

    point = all_scores(np.arange(len(y)))
    boots = pd.DataFrame([all_scores(i) for i in cluster_bootstrap(test, n_boot)])
    table = pd.DataFrame({"estimate": point, "ci_low": boots.quantile(0.025), "ci_high": boots.quantile(0.975)})
    table.loc["chance (prevalence)"] = np.average(y, weights=w)
    return table


def compare_models(test, finals, cv=None) -> pd.DataFrame:
    """Test PR-AUC and ROC AUC of every fitted model, next to its cross-validated PR-AUC."""
    y, w = test["distress"].to_numpy(), test["weight"].to_numpy()
    cv = pd.read_csv(RESULTS / "cv_summary.csv") if cv is None else cv
    cv = cv.set_index(["algorithm", "feature_set"])
    rows = []
    for (fs, algo), model in finals.items():
        p = model.predict_proba(test[columns(fs)])[:, 1]
        cv_pr = cv.loc[(algo, fs), "pr_auc_mean"] if (algo, fs) in cv.index else np.nan
        rows.append({"model": f"{NAMES[algo]} ({fs})", "test_pr_auc": average_precision_score(y, p, sample_weight=w),
                     "test_roc_auc": roc_auc_score(y, p, sample_weight=w), "cv_pr_auc": cv_pr})
    return pd.DataFrame(rows).sort_values("test_pr_auc", ascending=False)


def fairness(test, p_cal, threshold) -> pd.DataFrame:
    """Recall (share of distressed borrowers flagged) and AUC by group, against the overall figure."""
    y, w, pred = test["distress"].to_numpy(), test["weight"].to_numpy(), p_cal >= threshold
    overall = recall_score(y, pred, sample_weight=w)
    rows = []
    for col in ("sex", "residence", "age_group", "province"):
        for level, idx in test.groupby(col).indices.items():
            if len(idx) < 30 or y[idx].sum() == 0:
                continue
            r = recall_score(y[idx], pred[idx], sample_weight=w[idx])
            rows.append({"group": col, "level": level, "n": len(idx),
                         "observed_rate": np.average(y[idx], weights=w[idx]),
                         "mean_predicted": np.average(p_cal[idx], weights=w[idx]),
                         "recall": r, "recall_gap": r - overall,
                         "auc": roc_auc_score(y[idx], p_cal[idx], sample_weight=w[idx])})
    return pd.DataFrame(rows)


def county_holdout(train, algo, params) -> pd.DataFrame:
    """Train on some counties, test on unseen ones (5 groups of counties), vs. cluster-grouped folds."""
    X, y, w = train[columns("full")], train["distress"].to_numpy(), train["weight"].to_numpy()
    rows = []
    for scheme, groups in (("Unseen counties", train["county"]), ("Unseen clusters", train["cluster"])):
        for tr, va in GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, groups):
            p = make_model(algo, "full", params).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
            rows.append({"scheme": scheme, "pr_auc": average_precision_score(y[va], p, sample_weight=w[va]),
                         "chance": np.average(y[va], weights=w[va])})
    return pd.DataFrame(rows)


def plots(test, p_raw, p_cal, threshold, shap_imp):
    """Save the four test-set charts to Model/results/test_results.png."""
    y, w = test["distress"].to_numpy(), test["weight"].to_numpy()
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    prec, rec, _ = precision_recall_curve(y, p_raw, sample_weight=w)
    ax[0].plot(rec, prec, color=BLUE, lw=2)
    ax[0].axhline(np.average(y, weights=w), color=GREY, ls="--", label="Chance")
    ax[0].set(title="Precision-recall (test)", xlabel="Recall", ylabel="Precision", ylim=(0, 1))
    ax[0].legend(frameon=False)
    q = np.quantile(p_cal, np.linspace(0, 1, 11))
    b = np.clip(np.digitize(p_cal, q[1:-1]), 0, 9)
    keep = [i for i in range(10) if (b == i).any()]
    ax[1].plot([0, 1], [0, 1], color=GREY, ls="--")
    ax[1].plot([np.average(p_cal[b == i], weights=w[b == i]) for i in keep],
               [np.average(y[b == i], weights=w[b == i]) for i in keep], "o-", color=ORANGE, lw=2)
    ax[1].set(title="Calibration (test)", xlabel="Predicted risk (decile mean)", ylabel="Observed rate",
              xlim=(0, 1), ylim=(0, 1))
    cm = confusion_matrix(y, p_cal >= threshold, sample_weight=w, normalize="all")
    ax[2].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[2].text(j, i, f"{cm[i, j]:.1%}", ha="center", va="center", fontsize=12)
    ax[2].set(title=f"Confusion matrix (threshold {threshold:.2f})", xticks=[0, 1], yticks=[0, 1],
              xticklabels=["Not flagged", "Flagged"], yticklabels=["Not distressed", "Distressed"])
    s = shap_imp.sort_values()
    ax[3].barh(s.index, s.values, color=BLUE)
    ax[3].set(title="What drives the model (mean |SHAP|)", xlabel="Mean |SHAP| (log-odds)")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "test_results.png", dpi=150)
    return fig


def main():
    """Evaluate the saved model on the test set and save all results."""
    df = pd.read_parquet(DATA)
    train, test = df[~df["is_test"]].reset_index(drop=True), df[df["is_test"]].reset_index(drop=True)
    bundle = joblib.load(MODEL_DIR / "model.joblib")
    finals = joblib.load(RESULTS / "all_models.joblib")
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    p_raw = bundle["pipeline"].predict_proba(test[bundle["features"]])[:, 1]
    p_cal = predict_proba(bundle, test)

    metrics = test_metrics(test, p_raw, p_cal, bundle["threshold"])
    metrics.to_csv(RESULTS / "test_metrics.csv")
    print(metrics.round(3))
    compare_models(test, finals).to_csv(RESULTS / "test_all_models.csv", index=False)
    fairness(test, p_cal, bundle["threshold"]).to_csv(RESULTS / "fairness.csv", index=False)
    county_holdout(train, bundle["algorithm"], meta["params"]).to_csv(RESULTS / "county_holdout.csv", index=False)
    shap_imp = shap_by_group(bundle["pipeline"], train[bundle["features"]]).abs().mean()
    shap_imp.rename("mean_abs_shap").to_csv(RESULTS / "shap_importance.csv")
    plots(test, p_raw, p_cal, bundle["threshold"], shap_imp)

    meta["test"] = metrics["estimate"].to_dict()
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=1, default=float))
    print("Saved results to Model/results/")


if __name__ == "__main__":
    main()
