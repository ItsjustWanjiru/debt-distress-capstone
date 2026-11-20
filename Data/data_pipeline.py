"""Data pipeline: FinAccess 2024 raw file -> clean modelling table.

Steps
1. Load only the survey items the study needs from the Stata file (unzipped on first run).
2. Keep current digital credit borrowers with a valid answer to the repayment question (E2E).
3. Build the outcome: late / missed / no repayment AND a hardship sacrifice to repay.
4. Build the predictors (demographics, borrowing behaviour, controls).
5. Split 20% of the sampling clusters in each county into a test set.

Run from the project root:
    python -m Data.data_pipeline
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

DATA_DIR = Path(__file__).resolve().parent
RAW_FILE = DATA_DIR / "2024_Finaccess_Publicdata.dta"
RAW_ZIP = DATA_DIR / "2024_Finaccess_Publicdata.zip"       # compressed copy kept in the repository
OUT_FILE = DATA_DIR / "processed" / "model_data.parquet"
SEED = 2026

# Digital credit products (current use = code 1) and the suffix of their loan items.
PRODUCTS = {"mobile_bank": ("C1_15", "E"), "mobile_money": ("C1_16", "F"),
            "digital_app": ("C1_19", "I"), "hustler_fund": ("C1_23", "M")}
HARDSHIP_ITEMS = ["E2D__1", "E2D__3", "E2D__4", "E2D__7"]   # borrowed to repay, sold assets, cut food, repossessed
SAVINGS_ITEMS = ["C1_1a", "C1_1b"] + [f"C1_{i}" for i in range(2, 10)]
OTHER_LOAN_ITEMS = ["C1_13a", "C1_13b", "C1_13c", "C1_14", "C1_17", "C1_18", "C1_20", "C1_21", "C1_22"] + \
                   [f"C1_{i}" for i in range(24, 32)]
SHOCK_ITEMS = [f"R2A__{i}" for i in range(1, 13)]

COLUMNS = (["indWeight", "A07", "A08", "county", "A18", "Sex", "Education", "livelihoodcat", "NHM",
            "E2E", "E2A", "B3Ii", "B1Ii", "B1N2", "B2Ei", "B2Eii", "B2Eiii"]
           + [c for c, _ in PRODUCTS.values()]
           + [f"E1{x}{s}" for x in ("i", "ii", "iii", "xi") for _, s in PRODUCTS.values()]
           + [f"E2D__{i}" for i in range(1, 9)] + SAVINGS_ITEMS + OTHER_LOAN_ITEMS + SHOCK_ITEMS)

# Former provinces from KNBS county codes (1-47); used instead of 47 county dummies.
PROVINCE_BREAKS = [(6, "Coast"), (9, "North Eastern"), (17, "Eastern"), (22, "Central"), (36, "Rift Valley"),
                   (40, "Western"), (46, "Nyanza"), (47, "Nairobi")]


def province(code: int) -> str:
    """Former province for a KNBS county code (1 to 47)."""
    for upper, name in PROVINCE_BREAKS:
        if code <= upper:
            return name
    return "Nairobi"


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Read the needed columns from the FinAccess Stata file, extracting it from the zip on first run."""
    if not path.exists() and RAW_ZIP.exists():
        with zipfile.ZipFile(RAW_ZIP) as z:                       # the .dta is too large for GitHub unzipped
            z.extract(path.name, DATA_DIR)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: expected {RAW_ZIP} or {path}")
    df, _ = pyreadstat.read_dta(path, usecols=COLUMNS)
    return df.apply(pd.to_numeric, errors="coerce")       # some columns load as text


def purpose_group(code) -> str | float:
    """Group the reason for borrowing into four categories."""
    if pd.isna(code) or code in (98, 99):
        return np.nan
    code = int(code)
    if code in (4, 14, 15):
        return "day_to_day"
    if code == 1:
        return "emergency"
    if code in (2, 3, 5, 6, 7, 8, 10) or 18 <= code <= 24:
        return "productive"
    return "other"


def build_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Analytic sample, outcome and predictors."""
    uses = pd.DataFrame({p: raw[c] == 1 for p, (c, _) in PRODUCTS.items()})
    d = raw[uses.any(axis=1) & raw["E2E"].isin([1, 2, 3])].copy()
    uses = uses.loc[d.index]
    out = pd.DataFrame(index=d.index)

    # Survey design
    out["weight"] = d["indWeight"]
    out["cluster"] = d["A07"].astype(int)
    out["county"] = d["county"].astype(int)
    out["residence"] = d["A08"].map({1: "Rural", 2: "Urban"})
    out["province"] = out["county"].map(province)

    # Outcome: repayment problem AND a hardship sacrifice (using savings is not counted as hardship)
    out["late"] = d["E2E"].isin([1, 2]).astype(int)
    out["hardship"] = (d[HARDSHIP_ITEMS].fillna(0) == 1).any(axis=1).astype(int)
    out["distress"] = out["late"] * out["hardship"]
    out["worried"] = (d["B1N2"] == 1).astype(int)          # used only to check the outcome

    # Demographics
    out["age_group"] = pd.cut(d["A18"], [15, 24, 34, 44, 54, 120],
                              labels=["16-24", "25-34", "35-44", "45-54", "55+"]).astype(str)
    out["sex"] = d["Sex"].map({1: "Male", 2: "Female"})
    out["education"] = d["Education"].map({1: "None/other", 2: "Primary", 3: "Secondary", 4: "Tertiary",
                                           5: "None/other"})
    out["livelihood"] = d["livelihoodcat"].map({1: "Agriculture", 2: "Employed", 3: "Casual worker",
                                                4: "Own business", 5: "Dependent", 6: "Other"})
    out["household_size"] = d["NHM"]

    # Borrowing behaviour
    for p in PRODUCTS:
        out[f"uses_{p}"] = uses[p].astype(int)
    out["n_products"] = uses.sum(axis=1)
    # Loans in the past 12 months: missing for a product in use stays missing (imputed later).
    counts = pd.DataFrame({p: d[f"E1i{s}"].where(uses[p], 0) for p, (_, s) in PRODUCTS.items()})
    out["loans_12m"] = counts.sum(axis=1, min_count=4)
    # Outstanding loans are never recorded as 0, so a skipped count means none.
    # Counts above 20 (a few Fuliza values up to 900) are keying errors and set to missing.
    outstanding = pd.DataFrame({p: d[f"E1ii{s}"].fillna(0) for p, (_, s) in PRODUCTS.items()})
    out["outstanding_loans"] = outstanding.sum(axis=1).where(~(outstanding > 20).any(axis=1))
    income = d["B3Ii"].where(d["B3Ii"] > 0)
    out["repayment_burden"] = d["E2A"].where(~d["E2A"].isin([98, 99])) / income
    balance = sum(d[f"E1xi{s}"].fillna(0) for _, s in PRODUCTS.values())
    out["debt_to_income"] = balance / income
    # Purpose of the product with the most loans
    main = counts.fillna(0).where(uses, -1).idxmax(axis=1)   # only products actually in use
    purpose_code = pd.Series([d.at[i, f"E1iii{PRODUCTS[p][1]}"] for i, p in main.items()], index=d.index)
    out["purpose"] = purpose_code.map(purpose_group)

    # Controls
    out["shocks"] = d[SHOCK_ITEMS].sum(axis=1, min_count=1)
    out["savings_difficulty"] = d["B1Ii"].map({1: "Very difficult", 2: "Somewhat difficult",
                                               3: "Not difficult"}).fillna("Not asked")
    out["has_savings"] = (d[SAVINGS_ITEMS] == 1).any(axis=1).astype(int)
    literacy = (d["B2Ei"] == 1).astype(int) + (d["B2Eii"] == 3).astype(int) + (d["B2Eiii"] == 1).astype(int)
    out["literacy_score"] = literacy                          # "don't know" scores as incorrect
    out["nondigital_loans"] = (d[OTHER_LOAN_ITEMS] == 1).sum(axis=1)
    return out.reset_index(drop=True)


def cluster_split(df: pd.DataFrame, test_share: float = 0.2, seed: int = SEED) -> pd.Series:
    """Hold out ~20% of clusters within each county.

    Whole clusters go to one side so neighbours never appear in both sets. Out of 50 random
    splits, the one whose test distress rate is closest to the overall rate is kept.
    """
    rng = np.random.default_rng(seed)
    clusters = df.groupby("cluster")["county"].first()
    best, best_gap = None, np.inf
    for _ in range(50):
        test_clusters = []
        for _, ids in clusters.groupby(clusters):
            k = int(round(test_share * len(ids))) if len(ids) >= 3 else 0
            test_clusters += list(rng.choice(ids.index, size=k, replace=False))
        is_test = df["cluster"].isin(test_clusters)
        gap = abs(df.loc[is_test, "distress"].mean() - df["distress"].mean())
        if gap < best_gap:
            best, best_gap = is_test, gap
    return best


def main() -> pd.DataFrame:
    """Run the full data step and save the modelling table."""
    raw = load_raw()
    df = build_dataset(raw)
    df["is_test"] = cluster_split(df)
    OUT_FILE.parent.mkdir(exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)
    prev = np.average(df["distress"], weights=df["weight"])
    print(f"Respondents: {len(raw):,} | digital borrowers in sample: {len(df):,} | "
          f"weighted distress rate: {prev:.1%} | test rows: {df['is_test'].sum():,}")
    print(f"Saved {OUT_FILE}")
    return df


if __name__ == "__main__":
    main()
