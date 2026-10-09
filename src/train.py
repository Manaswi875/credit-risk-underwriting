"""Train and compare default models on an out-of-time split.

Train: 2012-2014 vintages.  Test: 2015 vintage (never seen during training).

Models compared:
  incumbent      LendingClub's own sub-grade (the benchmark to beat)
  logit_core     Logistic regression, core application fields
  logit_expanded Logistic regression, core + expanded credit attributes
  xgb_core       XGBoost, core application fields
  xgb_expanded   XGBoost, core + expanded credit attributes
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from xgboost import XGBClassifier

from prepare_data import EXPANDED

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "loans.parquet"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

CORE_NUM = ["fico", "dti", "log_annual_inc", "loan_amnt", "emp_length_yrs",
            "loan_to_income", "payment_to_income"]
CAT = ["home_ownership", "purpose", "verification_status"]
EXPANDED_NUM = [c for c in EXPANDED if c != "earliest_cr_line"] + ["credit_history_yrs"]

FEATURE_SETS = {
    "core": (CORE_NUM, CAT),
    "expanded": (CORE_NUM + EXPANDED_NUM, CAT),
}
NEAR_PRIME_FICO = 700  # LendingClub floor is ~660, so <700 is its near-prime slice


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y, p):
    auc = roc_auc_score(y, p)
    out = {"auc": auc, "gini": 2 * auc - 1, "ks": ks_stat(y, p)}
    if p.min() >= 0 and p.max() <= 1:
        out |= {"brier": brier_score_loss(y, p), "log_loss": log_loss(y, p)}
    return {k: round(float(v), 4) for k, v in out.items()}


def logit_pipeline(num, cat):
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            # Rank-to-normal transform: robust to the heavy tails in income/balances.
            ("scale", QuantileTransformer(output_distribution="normal",
                                          subsample=200_000, random_state=0)),
        ]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=200), cat),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(C=0.1, max_iter=2000))])


def xgb_frame(df, num, cat):
    X = df[num + cat].copy()
    for c in cat:
        X[c] = X[c].astype("category")
    return X


def main() -> None:
    df = pd.read_parquet(DATA)
    train = df[df["vintage"] <= 2014].reset_index(drop=True)
    test = df[df["vintage"] == 2015].reset_index(drop=True)
    y_tr, y_te = train["default"].values, test["default"].values
    near_prime = (test["fico"] < NEAR_PRIME_FICO).values
    print(f"train {len(train):,} ({y_tr.mean():.2%} default) | "
          f"test {len(test):,} ({y_te.mean():.2%} default)")

    preds = {"incumbent": test["sub_grade_rank"].values.astype(float)}

    for name, (num, cat) in FEATURE_SETS.items():
        # Logistic regression
        lr = logit_pipeline(num, cat).fit(train[num + cat], y_tr)
        preds[f"logit_{name}"] = lr.predict_proba(test[num + cat])[:, 1]
        joblib.dump(lr, MODELS / f"logit_{name}.joblib")

        # XGBoost, early-stopped on a random 10% of the training vintages
        X_tr = xgb_frame(train, num, cat)
        rng = np.random.default_rng(0)
        val = rng.random(len(X_tr)) < 0.1
        xgb = XGBClassifier(
            n_estimators=2000, learning_rate=0.03, max_depth=5,
            min_child_weight=50, subsample=0.8, colsample_bytree=0.7,
            reg_lambda=5.0, tree_method="hist", enable_categorical=True,
            eval_metric="auc", early_stopping_rounds=100, n_jobs=-1,
        )
        xgb.fit(X_tr[~val], y_tr[~val], eval_set=[(X_tr[val], y_tr[val])],
                verbose=False)
        preds[f"xgb_{name}"] = xgb.predict_proba(xgb_frame(test, num, cat))[:, 1]
        xgb.save_model(MODELS / f"xgb_{name}.json")
        print(f"  xgb_{name}: best iteration {xgb.best_iteration}")

    results = {}
    for name, p in preds.items():
        results[name] = {
            "all": metrics(y_te, p),
            "near_prime": metrics(y_te[near_prime], p[near_prime]),
        }
        print(f"{name:15s} AUC {results[name]['all']['auc']:.4f}  "
              f"KS {results[name]['all']['ks']:.4f}  "
              f"near-prime AUC {results[name]['near_prime']['auc']:.4f}")

    out = test[["default", "net_profit", "funded_amnt", "fico", "sub_grade",
                "int_rate", "addr_state"]].copy()
    for name, p in preds.items():
        out[f"score_{name}"] = p
    out.to_parquet(REPORTS / "test_predictions.parquet", index=False)

    results["_meta"] = {
        "train_vintages": "2012-2014", "test_vintage": 2015,
        "n_train": len(train), "n_test": len(test),
        "test_default_rate": round(float(y_te.mean()), 4),
        "n_test_near_prime": int(near_prime.sum()),
        "n_features": {k: len(v[0]) + len(v[1]) for k, v in FEATURE_SETS.items()},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    main()
