"""FastAPI service for the debt distress model.

    uvicorn API.main:app --port 8000        (from the project root)
    docker compose up --build               (same service in a container)

Endpoints: GET /health, POST /predict, POST /predict/batch. Interactive docs at /docs (the home page / redirects there).
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from Model.components import COUNTIES, predict_proba, profile_to_features, shap_by_group

MODEL_DIR = Path(__file__).resolve().parents[1] / "Model"
try:
    BUNDLE = joblib.load(MODEL_DIR / "model.joblib")
    META = json.loads((MODEL_DIR / "metadata.json").read_text())
except FileNotFoundError:
    BUNDLE, META = None, {}


@asynccontextmanager
async def lifespan(app):
    """Print the browser link when the server starts."""
    # Uvicorn prints 0.0.0.0, which browsers cannot open, so print the usable link (default port) as well
    logging.getLogger("uvicorn.error").info("Open http://localhost:8000 in your browser to use the API")
    yield


app = FastAPI(title="Debt Distress Risk API", version="1.0.0", lifespan=lifespan,
              description="Calibrated risk of debt distress for Kenyan digital credit borrowers (FinAccess 2024).")
NOTICE = "Group-level estimate for this type of borrower. Not for decisions about an individual loan applicant."


class Borrower(BaseModel):
    """A borrower profile. Unknown fields and out-of-range values are rejected (HTTP 422)."""

    model_config = ConfigDict(extra="forbid")
    age: int = Field(ge=16, le=110)
    sex: Literal["Female", "Male"]
    education: Literal["None/other", "Primary", "Secondary", "Tertiary"]
    livelihood: Literal["Agriculture", "Employed", "Casual worker", "Own business", "Dependent", "Other"]
    household_size: int = Field(ge=1, le=40)
    residence: Literal["Rural", "Urban"]
    county: Literal[tuple(COUNTIES)]  # type: ignore[valid-type]
    uses_mobile_bank: bool
    uses_mobile_money: bool
    uses_digital_app: bool
    uses_hustler_fund: bool
    loans_past_12_months: int | None = Field(default=None, ge=0, le=2000)
    outstanding_loans: int = Field(ge=0, le=50)
    monthly_income: float | None = Field(default=None, gt=0)
    monthly_repayment: float | None = Field(default=None, ge=0)
    outstanding_balance: float | None = Field(default=None, ge=0)
    main_purpose: Literal["day_to_day", "emergency", "productive", "other"]
    shocks_past_year: int = Field(ge=0, le=12)
    savings_difficulty: Literal["Very difficult", "Somewhat difficult", "Not difficult", "Not asked"]
    has_savings: bool
    literacy_score: int = Field(ge=0, le=3)
    nondigital_loans: int = Field(ge=0, le=17)

    @model_validator(mode="after")
    def digital_borrower(self):
        if not (self.uses_mobile_bank or self.uses_mobile_money or self.uses_digital_app or self.uses_hustler_fund):
            raise ValueError("The model covers digital credit borrowers: at least one product must be in use.")
        return self


class Batch(BaseModel):
    """A list of borrower profiles for batch scoring."""
    borrowers: list[Borrower] = Field(min_length=1, max_length=1000)


def score(borrowers: list[Borrower]) -> list[dict]:
    """Turn validated profiles into calibrated probabilities, risk flags and top drivers."""
    if BUNDLE is None:
        raise HTTPException(503, "Model not found. Run the training pipeline first.")
    X = profile_to_features([b.model_dump() for b in borrowers])
    prob = predict_proba(BUNDLE, X)
    drivers = shap_by_group(BUNDLE["pipeline"], X[BUNDLE["features"]])
    out = []
    for i in range(len(X)):
        top = drivers.iloc[i].sort_values(key=np.abs, ascending=False).head(3)
        out.append({"distress_probability": round(float(prob[i]), 4),
                    "high_risk": bool(prob[i] >= BUNDLE["threshold"]),
                    "top_drivers": [{"factor": k, "effect": "raises risk" if v > 0 else "lowers risk"}
                                    for k, v in top.items()],
                    "notice": NOTICE})
    return out


@app.get("/", include_in_schema=False)
def home():
    """Send visitors of the bare address to the interactive docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    """Report whether the model is loaded, with its name and threshold."""
    return {"status": "ok", "model_loaded": BUNDLE is not None, "model": META.get("model"),
            "threshold": META.get("threshold")}


@app.post("/predict")
def predict(borrower: Borrower):
    """Score one borrower profile."""
    return score([borrower])[0]


@app.post("/predict/batch")
def predict_batch(batch: Batch):
    """Score several borrower profiles in one request."""
    return {"results": score(batch.borrowers)}
