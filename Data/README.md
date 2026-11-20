# Data

**Source:** FinAccess 2024 Household Survey (Central Bank of Kenya, KNBS and FSD Kenya), public release.

**Pipeline:** `python -m Data.data_pipeline` (run from the project root) reads the file and writes `processed/model_data.parquet`:

1. Loads only the survey items the study needs.
2. Keeps current digital credit borrowers with a valid answer on loan repayment (7,213 people).
3. Builds the outcome: repaid late, missed a payment or did not repay **and** made a hardship sacrifice to repay (borrowed from elsewhere, sold assets, cut food spending, or had assets repossessed).
4. Builds the predictors and cleans survey codes (98/99 "don't know/refused", skip patterns, implausible counts).
5. Holds out 20% of the survey clusters in each county as a test set.
