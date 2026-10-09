"""Recalibrate predicted default probabilities on the most recent known vintage.

The 2015 vintage defaulted more often than 2012-2014 (14.9% vs. 13.3%), so the
main model under-predicts risk at the top end. Rank-ordering (who is riskier)
is unaffected; the probability *level* is what drifts, and that level matters
for pricing, loss forecasting, and reserve setting.

Approach (no peeking at the 2015 test vintage):
  fit      XGBoost (expanded features) on 2012-2013
  calibrate on 2014 - the most recent vintage a lender would have in hand
  evaluate on 2015

Calibrators compared: none, Platt scaling (logistic on the log-odds), isotonic.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze import BLUE, INK_2, ORANGE, AQUA  # noqa: E402  (also applies chart style)
from train import FEATURE_SETS, fit_xgb, xgb_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ece(y, p, bins=10):
    """Expected calibration error over equal-count score bins (in prob. points)."""
    dec = pd.qcut(p, bins, labels=False, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "b": dec}).groupby("b")
    return float((g.size() / len(y) * (g["p"].mean() - g["y"].mean()).abs()).sum())


def summarize(y, p):
    return {
        "auc": round(roc_auc_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 5),
        "log_loss": round(log_loss(y, p), 5),
        "ece_pct_pts": round(ece(y, p) * 100, 3),
        "mean_predicted": round(float(p.mean()), 4),
        "observed_rate": round(float(y.mean()), 4),
        "top_decile_predicted": round(float(np.sort(p)[-len(p) // 10:].mean()), 4),
        "top_decile_observed": round(float(y[np.argsort(p)[-len(p) // 10:]].mean()), 4),
    }


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "processed" / "loans.parquet")
    num, cat = FEATURE_SETS["expanded"]
    fit_df = df[df["vintage"] <= 2013]
    cal_df = df[df["vintage"] == 2014]
    test_df = df[df["vintage"] == 2015]

    model = fit_xgb(fit_df, num, cat)
    p_cal = model.predict_proba(xgb_frame(cal_df, num, cat))[:, 1]
    p_test = model.predict_proba(xgb_frame(test_df, num, cat))[:, 1]
    y_cal, y_test = cal_df["default"].values, test_df["default"].values

    platt = LogisticRegression(C=1e6).fit(logit(p_cal)[:, None], y_cal)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p_cal, y_cal)

    # The production model (trained 2012-2014, uncalibrated) for reference.
    main_preds = pd.read_parquet(REPORTS / "test_predictions.parquet")["score_xgb_expanded"].values

    variants = {
        "main_model_uncalibrated": main_preds,
        "uncalibrated_2012_13": p_test,
        "platt_on_2014": platt.predict_proba(logit(p_test)[:, None])[:, 1],
        "isotonic_on_2014": iso.predict(p_test),
    }
    results = {k: summarize(y_test, v) for k, v in variants.items()}
    results["_meta"] = {"fit": "2012-2013", "calibrate": "2014", "test": "2015",
                        "platt_slope": round(float(platt.coef_[0, 0]), 4),
                        "platt_intercept": round(float(platt.intercept_[0]), 4)}
    (REPORTS / "calibration.json").write_text(json.dumps(results, indent=2))
    for k, v in results.items():
        print(k, v)

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for key, label, color in [("uncalibrated_2012_13", "Before: uncalibrated", ORANGE),
                              ("platt_on_2014", "After: Platt scaling on 2014", BLUE),
                              ("isotonic_on_2014", "After: isotonic on 2014", AQUA)]:
        p = variants[key]
        g = pd.DataFrame({"p": p, "y": y_test, "b": pd.qcut(p, 10, labels=False)}) \
            .groupby("b").mean()
        ax.plot(g["p"] * 100, g["y"] * 100, "o-", color=color, label=label,
                markersize=6, markeredgecolor="white", markeredgewidth=1.5)
    lim = [0, 40]
    ax.plot(lim, lim, "--", color=INK_2, linewidth=1, label="Perfect calibration")
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("Predicted default rate (%)")
    ax.set_ylabel("Observed default rate, 2015 (%)")
    ax.set_title("Recalibration on the latest vintage")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_calibration.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
