"""Translate model scores into underwriting decisions and business impact.

Everything here is computed on the 2015 out-of-time test vintage.

Outputs (reports/):
  business_impact.json     headline numbers used in the README
  fig_tradeoff.png         approval rate vs. default rate
  fig_profit.png           realized profit per applicant vs. approval rate
  fig_auc.png              AUC by model, all applicants vs. near-prime
  fig_calibration.png      predicted vs. observed default by decile
  fig_drivers.png          top risk drivers (SHAP) for the expanded model
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from train import CAT, FEATURE_SETS, NEAR_PRIME_FICO, xgb_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PER = 100_000  # express impact per 100K applicants

# Colorblind-validated categorical slots (light mode), in fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {  # score column -> (label, color)
    "score_incumbent": ("LendingClub grade (benchmark)", ORANGE),
    "score_xgb_core": ("XGBoost - core fields", AQUA),
    "score_xgb_expanded": ("XGBoost - core + expanded data", BLUE),
}

LABELS = {
    "payment_to_income": "Loan payment / income",
    "fico": "FICO score",
    "acc_open_past_24mths": "Accounts opened, last 24 mo",
    "dti": "Debt-to-income",
    "home_ownership": "Home ownership",
    "percent_bc_gt_75": "% of cards over 75% utilized",
    "mths_since_recent_bc": "Months since newest card",
    "purpose": "Loan purpose",
    "loan_to_income": "Loan amount / income",
    "mo_sin_old_rev_tl_op": "Age of oldest revolving acct",
    "log_annual_inc": "Annual income",
    "total_acc": "Total credit accounts",
    "mths_since_recent_inq": "Months since last inquiry",
    "bc_open_to_buy": "Available card credit",
    "tot_hi_cred_lim": "Total credit limit",
    "revol_util": "Revolving utilization",
    "mort_acc": "Mortgage accounts",
    "credit_history_yrs": "Credit history length",
}

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "legend.frameon": False,
    "lines.linewidth": 2,
})


def policy_curve(df, score, grid):
    """Approve the lowest-risk fraction `a` of applicants for each a in grid."""
    rng = np.random.default_rng(0)
    # Random tie-break so a coarse score (35 grades) isn't ordered by row position.
    order = np.lexsort((rng.random(len(df)), df[score].values))
    d = df["default"].values[order]
    profit = df["net_profit"].values[order]
    n = len(df)
    rows = []
    for a in grid:
        k = int(round(a * n))
        rows.append({"approval_rate": a, "default_rate": d[:k].mean(),
                     "profit_per_applicant": profit[:k].sum() / n})
    return pd.DataFrame(rows)


def max_approval_at_bad_rate(curve, bad_rate):
    ok = curve[curve["default_rate"] <= bad_rate]
    return float(ok["approval_rate"].max())


def impact(df, base="score_incumbent", challenger="score_xgb_expanded",
           approval=0.80):
    grid = np.round(np.arange(0.30, 1.0001, 0.0025), 4)
    cb, cc = policy_curve(df, base, grid), policy_curve(df, challenger, grid)
    at = lambda c: c.loc[np.isclose(c["approval_rate"], approval)].iloc[0]
    b, c = at(cb), at(cc)
    same_risk_approval = max_approval_at_bad_rate(cc, b["default_rate"])
    return {
        "approval_rate_policy": approval,
        "benchmark_default_rate": round(b["default_rate"], 4),
        "model_default_rate": round(c["default_rate"], 4),
        "relative_default_reduction": round(1 - c["default_rate"] / b["default_rate"], 4),
        "defaults_avoided_per_100k": round((b["default_rate"] - c["default_rate"]) * approval * PER),
        "profit_lift_per_100k_usd": round((c["profit_per_applicant"] - b["profit_per_applicant"]) * PER),
        "same_risk_model_approval_rate": same_risk_approval,
        "extra_approvals_per_100k_at_same_risk": round((same_risk_approval - approval) * PER),
    }, cb, cc


def fig_tradeoff(df):
    grid = np.round(np.arange(0.30, 1.0001, 0.01), 4)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for col, (label, color) in SERIES.items():
        c = policy_curve(df, col, grid)
        ax.plot(c["approval_rate"] * 100, c["default_rate"] * 100, color=color, label=label)
    ax.set_xlabel("Approval rate (%)  - approving the lowest-risk applicants first")
    ax.set_ylabel("Default rate of approved loans (%)")
    ax.set_title("Same approval rate, fewer defaults")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_tradeoff.png")
    plt.close(fig)


def fig_profit(df):
    grid = np.round(np.arange(0.30, 1.0001, 0.01), 4)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for col, (label, color) in SERIES.items():
        c = policy_curve(df, col, grid)
        ax.plot(c["approval_rate"] * 100, c["profit_per_applicant"], color=color, label=label)
    ax.set_xlabel("Approval rate (%)")
    ax.set_ylabel("Realized net profit per applicant ($)")
    ax.set_title("Better risk ranking earns more at every approval rate")
    ax.legend(loc="lower center")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_profit.png")
    plt.close(fig)


def fig_auc(metrics):
    names = [("incumbent", "LendingClub grade"), ("logit_core", "Logistic - core"),
             ("xgb_core", "XGBoost - core"), ("logit_expanded", "Logistic - expanded"),
             ("xgb_expanded", "XGBoost - expanded")]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for seg, color, off, label in [("all", BLUE, 0.17, "All applicants"),
                                   ("near_prime", ORANGE, -0.17, f"Near-prime (FICO < {NEAR_PRIME_FICO})")]:
        vals = [metrics[n][seg]["auc"] for n, _ in names]
        ax.barh(y + off, vals, height=0.3, color=color, label=label)
        for yi, v in zip(y + off, vals):
            ax.text(v + 0.001, yi, f"{v:.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, [n for _, n in names])
    ax.set_xlim(0.62, 0.71)
    ax.set_xlabel("AUC on 2015 out-of-time test (higher = better ranking of risk)")
    ax.set_title("The data matters more than the algorithm")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_auc.png")
    plt.close(fig)


def fig_calibration(df):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for col, label, color in [("score_xgb_expanded", "XGBoost - expanded", BLUE),
                              ("score_logit_expanded", "Logistic - expanded", ORANGE)]:
        dec = pd.qcut(df[col], 10, labels=False)
        g = df.groupby(dec).agg(pred=(col, "mean"), obs=("default", "mean"))
        ax.plot(g["pred"] * 100, g["obs"] * 100, "o-", color=color, label=label,
                markersize=6, markeredgecolor="white", markeredgewidth=1.5)
    lim = [0, 45]
    ax.plot(lim, lim, "--", color=INK_2, linewidth=1, label="Perfect calibration")
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("Predicted default rate (%)")
    ax.set_ylabel("Observed default rate (%)")
    ax.set_title("Calibration by score decile (2015)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_calibration.png")
    plt.close(fig)


def fig_drivers(test_features):
    """Mean |SHAP| for the expanded XGBoost model via native tree contributions."""
    model = xgb.XGBClassifier()
    model.load_model(ROOT / "models" / "xgb_expanded.json")
    num, cat = FEATURE_SETS["expanded"]
    X = xgb_frame(test_features.sample(20_000, random_state=0), num, cat)
    contrib = model.get_booster().predict(
        xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=X.columns).sort_values()
    top = imp.tail(12)
    top.index = [LABELS.get(f, f) for f in top.index]
    expanded = set(FEATURE_SETS["expanded"][0]) - set(FEATURE_SETS["core"][0])
    colors = [BLUE if f in expanded else ORANGE for f in imp.tail(12).index]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(top.index, top.values, color=colors, height=0.6)
    ax.set_xlabel("Mean |SHAP value| (impact on log-odds of default)")
    ax.set_title("Top risk drivers - expanded model")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="Core application field"),
                       Patch(color=BLUE, label="Expanded credit attribute")],
              loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_drivers.png")
    plt.close(fig)
    return imp.sort_values(ascending=False).round(4).to_dict()


def main() -> None:
    preds = pd.read_parquet(REPORTS / "test_predictions.parquet")
    metrics = json.loads((REPORTS / "metrics.json").read_text())

    overall, _, _ = impact(preds)
    near = preds[preds["fico"] < NEAR_PRIME_FICO]
    near_prime, _, _ = impact(near)
    data_only, _, _ = impact(preds, base="score_xgb_core")

    # Best-profit policy for each scorer.
    grid = np.round(np.arange(0.30, 1.0001, 0.0025), 4)
    best_policy = {}
    for col in SERIES:
        c = policy_curve(preds, col, grid)
        row = c.loc[c["profit_per_applicant"].idxmax()]
        best_policy[col.removeprefix("score_")] = {
            "approval_rate": float(row["approval_rate"]),
            "default_rate": round(float(row["default_rate"]), 4),
            "profit_per_100k_usd": round(float(row["profit_per_applicant"]) * PER),
        }

    fig_tradeoff(preds)
    fig_profit(preds)
    fig_auc(metrics)
    fig_calibration(preds)

    df = pd.read_parquet(ROOT / "data" / "processed" / "loans.parquet")
    drivers = fig_drivers(df[df["vintage"] == 2015])

    out = {
        "vs_lendingclub_grade_all": overall,
        "vs_lendingclub_grade_near_prime": near_prime,
        "value_of_expanded_data_vs_core_xgb": data_only,
        "profit_maximizing_policy": best_policy,
        "top_drivers_mean_abs_shap": dict(list(drivers.items())[:15]),
    }
    (REPORTS / "business_impact.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "top_drivers_mean_abs_shap"},
                     indent=2))
    print("top drivers:", list(drivers)[:10])


if __name__ == "__main__":
    main()
