"""Fair-lending review of the underwriting models (2015 test vintage).

LendingClub data has no applicant race, ethnicity, or sex, and none of those are
model inputs. To test for disparate impact we use an *area-level* proxy, the
standard approach in redlining reviews: each loan's 3-digit ZIP is joined to
Census ACS 2017-2021 demographics, and areas are grouped by minority population
share and by income.

Checks:
  1. Adverse impact ratio (AIR) of approval rates under an 80%-approval policy,
     for the incumbent grade and both XGBoost models (four-fifths rule: AIR >= 0.80).
  2. Differential prediction: does the model over- or under-predict default for
     any group? (predicted vs. observed default rate, AUC within group)
  3. Proxy screen: how strongly each model input tracks area minority share.
  4. Adverse-action reason codes: the top SHAP-based reasons a declined
     applicant would receive (ECOA / Regulation B requires specific reasons).
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze import BLUE, INK_2, LABELS, SERIES  # noqa: E402
from train import FEATURE_SETS, xgb_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
REPORTS = ROOT / "reports"
APPROVAL = 0.80
ACS_URL = ("https://www2.census.gov/programs-surveys/acs/summary_file/2021/"
           "table-based-SF/data/5YRData/acsdt5y2021-{t}.dat")


def load_zip3_demographics() -> tuple[pd.DataFrame, float]:
    """Aggregate ZCTA-level ACS tables to 3-digit ZIP areas."""
    race = pd.read_csv(EXT / "acsdt5y2021-b03002.dat", sep="|",
                       usecols=["GEO_ID", "B03002_E001", "B03002_E003"])
    inc = pd.read_csv(EXT / "acsdt5y2021-b19013.dat", sep="|",
                      usecols=["GEO_ID", "B19013_E001"])
    national_median = float(inc.loc[inc["GEO_ID"] == "0100000US", "B19013_E001"].iloc[0])

    z = race.merge(inc, on="GEO_ID")
    z = z[z["GEO_ID"].str.startswith("860Z200US")].copy()
    z["zip3"] = z["GEO_ID"].str[-5:].str[:3]
    z = z.rename(columns={"B03002_E001": "pop", "B03002_E003": "nh_white",
                          "B19013_E001": "med_inc"})
    z["med_inc"] = z["med_inc"].where(z["med_inc"] > 0)  # suppressed values are negative
    z["inc_x_pop"] = z["med_inc"] * z["pop"]
    z["pop_inc"] = z["pop"].where(z["med_inc"].notna(), 0)
    g = z.groupby("zip3")[["pop", "nh_white", "inc_x_pop", "pop_inc"]].sum()
    g = g[g["pop"] > 0]
    out = pd.DataFrame({
        "minority_share": 1 - g["nh_white"] / g["pop"],
        # Population-weighted mean of ZCTA medians: an approximation of the area median.
        "area_median_income": g["inc_x_pop"] / g["pop_inc"].replace(0, np.nan),
    })
    return out.reset_index(), national_median


def approve_mask(score, approval=APPROVAL):
    """One portfolio-wide cutoff: approve the lowest-risk `approval` share."""
    rng = np.random.default_rng(0)
    order = np.lexsort((rng.random(len(score)), score))
    approved = np.zeros(len(score), bool)
    approved[order[: int(round(approval * len(score)))]] = True
    return approved


def group_table(df, group_col, reference):
    rows = []
    for col in SERIES:
        approved = df[f"approved_{col}"]
        rates = df.assign(a=approved).groupby(group_col, observed=True)["a"].mean()
        for g, r in rates.items():
            rows.append({"scorer": col.removeprefix("score_"), "group": str(g),
                         "approval_rate": round(r, 4),
                         "air_vs_reference": round(r / rates[reference], 4)})
    return pd.DataFrame(rows)


def differential_prediction(df, group_col):
    rows = []
    for g, d in df.groupby(group_col, observed=True):
        rows.append({
            "group": str(g), "loans": len(d),
            "observed_default": round(d["default"].mean(), 4),
            "predicted_default": round(d["score_xgb_expanded"].mean(), 4),
            "auc": round(roc_auc_score(d["default"], d["score_xgb_expanded"]), 4),
        })
    return pd.DataFrame(rows)


def reason_codes(loans_2015, preds, approved, n=20_000):
    """Top-2 risk-increasing SHAP features for a sample of declined applicants."""
    model = xgb.XGBClassifier()
    model.load_model(ROOT / "models" / "xgb_expanded.json")
    num, cat = FEATURE_SETS["expanded"]
    declined_idx = np.flatnonzero(~approved)
    idx = np.random.default_rng(0).choice(declined_idx, min(n, len(declined_idx)),
                                          replace=False)
    X = xgb_frame(loans_2015.iloc[idx], num, cat)
    contrib = model.get_booster().predict(
        xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)[:, :-1]
    top2 = np.argsort(-contrib, axis=1)[:, :2]
    names = np.array([LABELS.get(c, c) for c in X.columns])
    principal = pd.Series(names[top2[:, 0]]).value_counts(normalize=True)
    any_top2 = pd.Series(names[top2].ravel()).value_counts() / len(idx)
    return principal, any_top2


def fig_fairness(by_quartile, by_lmi):
    """Adverse impact ratio by group (dots: non-zero baseline is fine)."""
    groups = [("Q2", by_quartile), ("Q3", by_quartile), ("Q4 (highest)", by_quartile),
              ("Low/moderate-income area", by_lmi)]
    ticks = ["Minority share\nQ2", "Minority share\nQ3", "Minority share\nQ4 (highest)",
             "Low/moderate-\nincome areas"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(groups))
    for i, (col, (label, color)) in enumerate(SERIES.items()):
        scorer = col.removeprefix("score_")
        vals = [t[(t["scorer"] == scorer) & (t["group"] == g)]["air_vs_reference"].iloc[0]
                for g, t in groups]
        ax.plot(x + (i - 1) * 0.12, vals, "o", color=color, label=label, markersize=9,
                markeredgecolor="white", markeredgewidth=2)
    ax.axhline(0.80, color="#e34948", linewidth=1.5, linestyle="--")
    ax.text(len(groups) - 0.5, 0.81, "four-fifths rule threshold (0.80)", ha="right",
            fontsize=8, color=INK_2)
    ax.axhline(1.0, color=INK_2, linewidth=1)
    ax.text(len(groups) - 0.5, 1.007, "parity with reference group", ha="right",
            fontsize=8, color=INK_2)
    ax.set_xticks(x, ticks, fontsize=9)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(0.75, 1.05)
    ax.set_ylabel("Adverse impact ratio (approval rate\nvs. reference group)")
    ax.set_title("All groups clear the four-fifths rule at 80% approval")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_fairness.png")
    plt.close(fig)


def fig_reasons(principal):
    top = principal.head(8).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(top.index, top.values * 100, color=BLUE, height=0.6)
    for y, v in enumerate(top.values * 100):
        ax.text(v + 0.4, y, f"{v:.0f}%", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("Share of declined applicants with this as #1 reason (%)")
    ax.set_title("Top reason for decline (adverse-action reason codes)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_reason_codes.png")
    plt.close(fig)


def main() -> None:
    for t in ("b03002", "b19013"):
        if not (EXT / f"acsdt5y2021-{t}.dat").exists():
            raise SystemExit(f"Missing Census file; download:\n  {ACS_URL.format(t=t)}"
                             f"\n  into {EXT}/")
    demo, national_median = load_zip3_demographics()

    preds = pd.read_parquet(REPORTS / "test_predictions.parquet")
    preds["zip3"] = preds["zip_code"].str[:3]
    for col in SERIES:
        preds[f"approved_{col}"] = approve_mask(preds[col].values)
    df = preds.merge(demo, on="zip3", how="left")
    matched = df["minority_share"].notna()
    df = df[matched].copy()

    # Quartiles are over loans, so each group holds ~25% of applicants.
    df["minority_q"] = pd.qcut(df["minority_share"], 4,
                               labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    df["majority_minority"] = np.where(df["minority_share"] >= 0.5,
                                       "Majority-minority area", "Other areas")
    df["lmi_area"] = np.where(df["area_median_income"] < 0.8 * national_median,
                              "Low/moderate-income area", "Other areas")

    by_quartile = group_table(df, "minority_q", "Q1 (lowest)")
    by_majmin = group_table(df, "majority_minority", "Other areas")
    by_lmi = group_table(df, "lmi_area", "Other areas")
    diffpred = pd.concat([
        differential_prediction(df, "minority_q").assign(dimension="minority quartile"),
        differential_prediction(df, "majority_minority").assign(dimension="majority-minority"),
        differential_prediction(df, "lmi_area").assign(dimension="LMI area"),
    ])

    # Proxy screen on the expanded model's numeric inputs.
    loans = pd.read_parquet(ROOT / "data" / "processed" / "loans.parquet")
    loans_2015 = loans[loans["vintage"] == 2015].reset_index(drop=True)
    loans_2015["zip3"] = loans_2015["zip_code"].str[:3]
    proxy_df = loans_2015.merge(demo, on="zip3", how="left").dropna(subset=["minority_share"])
    num = FEATURE_SETS["expanded"][0]
    proxy = proxy_df[num].corrwith(proxy_df["minority_share"], method="spearman") \
        .abs().sort_values(ascending=False).round(3)

    approved_main = approve_mask(preds["score_xgb_expanded"].values)
    principal, any_top2 = reason_codes(loans_2015, preds, approved_main)

    fig_fairness(by_quartile, by_lmi)
    fig_reasons(principal)

    out = {
        "_meta": {
            "policy": f"approve lowest-risk {APPROVAL:.0%} (single portfolio cutoff)",
            "demographics": "Census ACS 2017-2021 5-year, aggregated ZCTA -> 3-digit ZIP",
            "national_median_household_income": national_median,
            "loans_matched_to_area": int(matched.sum()),
            "loans_unmatched": int((~matched).sum()),
            "share_in_majority_minority_areas": round(float((df["majority_minority"] == "Majority-minority area").mean()), 4),
            "share_in_lmi_areas": round(float((df["lmi_area"] == "Low/moderate-income area").mean()), 4),
            "minority_share_quartile_edges": [round(float(v), 3) for v in
                                              df["minority_share"].quantile([0, .25, .5, .75, 1])],
        },
        "air_by_minority_quartile": by_quartile.to_dict("records"),
        "air_majority_minority": by_majmin.to_dict("records"),
        "air_lmi_area": by_lmi.to_dict("records"),
        "differential_prediction_xgb_expanded": diffpred.to_dict("records"),
        "proxy_screen_abs_spearman_top10": proxy.head(10).to_dict(),
        "reason_codes_principal_share": principal.head(10).round(4).to_dict(),
        "reason_codes_in_top2_share": any_top2.head(10).round(4).to_dict(),
    }
    (REPORTS / "fair_lending.json").write_text(json.dumps(out, indent=2, default=str))

    pd.set_option("display.width", 140)
    print(json.dumps(out["_meta"], indent=2))
    for name, t in [("minority quartile", by_quartile), ("majority-minority", by_majmin),
                    ("LMI", by_lmi)]:
        print(f"\n== {name}\n", t.pivot(index="group", columns="scorer",
                                         values="air_vs_reference"))
        print(t.pivot(index="group", columns="scorer", values="approval_rate"))
    print("\n== differential prediction\n", diffpred)
    print("\n== proxy screen\n", proxy.head(10))
    print("\n== principal reasons\n", principal.head(8).round(3))


if __name__ == "__main__":
    main()
