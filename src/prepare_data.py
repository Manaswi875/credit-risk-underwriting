"""Build a clean, leakage-free modeling table from the raw LendingClub file.

Population: 36-month loans issued 2012-2015 that have reached a final outcome
(Fully Paid or Charged Off). Restricting to matured vintages avoids the bias of
keeping only loans that ended early (which skews toward defaults and prepayments).

Only fields known at application time are kept. Post-origination fields
(payments, recoveries, last FICO, etc.) are used solely to compute the realized
profit of each loan for the business analysis, never as model inputs.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
OUT = ROOT / "data" / "processed" / "loans.parquet"

# Core application fields: what a traditional score-driven underwriter sees.
CORE = [
    "fico_range_low", "fico_range_high", "dti", "annual_inc", "loan_amnt",
    "emp_length", "home_ownership", "purpose", "verification_status",
    "installment",
]

# Expanded credit-file attributes beyond the score: tradeline depth, utilization,
# recency of credit-seeking, payment behavior. Stand-in for "new data sources".
EXPANDED = [
    "revol_util", "revol_bal", "bc_util", "bc_open_to_buy", "total_rev_hi_lim",
    "tot_cur_bal", "avg_cur_bal", "tot_hi_cred_lim", "total_bc_limit",
    "total_il_high_credit_limit", "open_acc", "total_acc", "mort_acc",
    "num_actv_rev_tl", "num_rev_tl_bal_gt_0", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_sats", "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "acc_open_past_24mths", "inq_last_6mths", "mths_since_recent_inq",
    "mo_sin_old_rev_tl_op", "mo_sin_old_il_acct", "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl", "mths_since_recent_bc", "delinq_2yrs",
    "mths_since_last_delinq", "num_accts_ever_120_pd", "num_tl_90g_dpd_24m",
    "pub_rec", "pub_rec_bankruptcies", "tax_liens", "collections_12_mths_ex_med",
    "earliest_cr_line",
]

# LendingClub's own risk grade: the incumbent model we benchmark against.
INCUMBENT = ["sub_grade", "int_rate"]

# Outcome / profit / geography fields (NOT model inputs). zip_code (3-digit) is
# kept only to join area demographics for the fair-lending review.
OUTCOME = ["loan_status", "issue_d", "term", "funded_amnt", "total_pymnt",
           "addr_state", "zip_code"]


def main() -> None:
    cols = list(dict.fromkeys(CORE + EXPANDED + INCUMBENT + OUTCOME))
    df = pd.read_csv(RAW, usecols=cols, low_memory=False)

    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df = df[
        (df["term"].str.strip() == "36 months")
        & df["issue_d"].dt.year.between(2012, 2015)
        & df["loan_status"].isin(["Fully Paid", "Charged Off"])
    ].copy()

    df["default"] = (df["loan_status"] == "Charged Off").astype(int)
    df["vintage"] = df["issue_d"].dt.year
    # Realized lifetime return per loan (cash collected vs. cash lent).
    df["net_profit"] = df["total_pymnt"] - df["funded_amnt"]

    # Derived application-time features.
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["emp_length_yrs"] = (
        df["emp_length"].str.extract(r"(\d+)").astype(float)[0]
        .where(~df["emp_length"].str.contains("< 1", na=False), 0.5)
    )
    df["log_annual_inc"] = np.log1p(df["annual_inc"])
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)
    df["payment_to_income"] = df["installment"] * 12 / df["annual_inc"].replace(0, np.nan)
    df["credit_history_yrs"] = (
        df["issue_d"] - pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    ).dt.days / 365.25
    df["int_rate"] = df["int_rate"].astype(float)
    df["sub_grade_rank"] = df["sub_grade"].rank(method="dense")  # A1=1 ... G5=35

    df = df.drop(columns=["fico_range_low", "fico_range_high", "emp_length",
                          "earliest_cr_line", "term", "total_pymnt"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    summary = df.groupby("vintage").agg(loans=("default", "size"),
                                        default_rate=("default", "mean"))
    print(summary.round(4))
    print(f"Saved {len(df):,} loans x {df.shape[1]} cols -> {OUT}")


if __name__ == "__main__":
    main()
