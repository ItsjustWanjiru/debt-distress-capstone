"""Batch prediction: score a CSV of borrower profiles with the saved model.

The CSV needs the model's feature columns (listed in Model/metadata.json).
    python -m Model.predict --input Model/example_input.csv --output predictions.csv
"""

import argparse

import joblib
import pandas as pd

from Model.components import predict_proba
from Model.train import MODEL_DIR


def main():
    """Score a CSV of borrower profiles with the saved model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(MODEL_DIR / "example_input.csv"))
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    bundle = joblib.load(MODEL_DIR / "model.joblib")
    X = pd.read_csv(args.input)
    missing = set(bundle["features"]) - set(X.columns)
    if missing:
        raise SystemExit(f"Input is missing columns: {sorted(missing)}")
    X["distress_probability"] = predict_proba(bundle, X).round(4)
    X["flag"] = (X["distress_probability"] >= bundle["threshold"]).astype(int)
    X.to_csv(args.output, index=False)
    print(f"Scored {len(X)} profiles -> {args.output}")


if __name__ == "__main__":
    main()
