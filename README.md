# Credit Risk Underwriting: Does More Data Beat a Better Algorithm?

Default-prediction models for unsecured personal loans, built on **589K LendingClub loans (2012–2015)** and judged the way a lender would: on an **out-of-time** test set, against the **incumbent risk grade**, and in terms of **approvals, losses, and profit** rather than accuracy alone.

## Executive summary

**Question:** For a consumer lender serving near-prime borrowers, is it worth adding credit-file attributes beyond the score (utilization patterns, recent credit-seeking, account age) to the underwriting model?

**Answer: yes. The new data drives the gain, not the algorithm.**

| | LendingClub grade (benchmark) | Best model (XGBoost + expanded data) |
|---|---|---|
| AUC, all applicants | 0.679 | **0.698** |
| AUC, near-prime (FICO < 700) | 0.643 | **0.667** |
| Default rate at 80% approval | 11.9% | **11.4%** (−4.3% relative) |
| Approval rate at the *same* default rate | 80.0% | **83.75%** |

Per 100,000 applicants, on loans the models never saw (2015 vintage):

- **+3,750 more approvals at the same risk level**, or +4,500 within the near-prime segment.
- **412 fewer defaults** at an unchanged 80% approval rate.
- **+$5.8M realized net profit** at an unchanged 80% approval rate (see the caveats below).

With only the core application fields, neither logistic regression nor XGBoost beats LendingClub's own grade (AUC 0.676–0.677 vs. 0.679). Adding the expanded attributes lifts both algorithms by about 0.02 AUC. The gain is largest for near-prime applicants, where a FICO score alone says the least.

![Approval vs default tradeoff](reports/fig_tradeoff.png)

## Approach

**Population.** 36-month loans issued 2012–2015 that reached a final outcome (Fully Paid or Charged Off). Only fully matured vintages are used. Keeping recent vintages would keep only the loans that ended *early*, which over-represents defaults and prepayments.

**Out-of-time split.** Train on the 2012–2014 vintages (306K loans, 13.3% default rate). Test on the 2015 vintage (283K loans, 14.9% default rate). This mirrors real deployment: a model built on past loans has to perform on future ones, even as the credit mix shifts.

**No leakage.** Only fields known at application time are model inputs. Post-origination fields (payments, recoveries, updated FICO) are used only to measure each loan's realized profit.

**Feature sets.**
- *Core (10 features):* FICO, DTI, income, loan amount, employment length, payment-to-income, loan-to-income, home ownership, purpose, income verification. This is roughly what a score-driven underwriter sees.
- *Expanded (48 features):* core plus 38 credit-file attributes, including revolving and bankcard utilization, accounts opened recently, months since the last inquiry and the newest card, account age, delinquency history, and limits and balances. These stand in for evaluating a new data source.

**Models.** Logistic regression (quantile-normalized, L2-regularized), the industry-standard scorecard baseline, and XGBoost (early-stopped). Both are fit on each feature set. LendingClub's sub-grade, its own priced risk model, is the benchmark.

**Business translation.** Rank applicants by predicted risk and approve the lowest-risk X%. Then measure the default rate and realized profit (total payments − amount funded) of the approved book, and compare policies at equal approval rate and at equal default rate.

## Results

| Model | AUC | KS | Near-prime AUC | Brier |
|---|---|---|---|---|
| LendingClub grade | 0.679 | 0.262 | 0.643 | – |
| Logistic, core | 0.677 | 0.261 | 0.642 | 0.1216 |
| XGBoost, core | 0.676 | 0.254 | 0.641 | 0.1211 |
| Logistic, expanded | 0.694 | 0.285 | 0.663 | 0.1197 |
| **XGBoost, expanded** | **0.698** | **0.285** | **0.667** | **0.1188** |

![AUC by model](reports/fig_auc.png)

**What drives risk.** Payment burden relative to income and the FICO score lead. The third-strongest driver, ahead of DTI, is an expanded attribute: **number of accounts opened in the last 24 months**, a signal of recent credit-seeking. Card utilization concentration (share of cards over 75% utilized) and time since the newest card also rank in the top 7.

![Top risk drivers](reports/fig_drivers.png)

**Profit.** At every approval rate, better risk ranking earns more realized profit per applicant.

![Profit vs approval rate](reports/fig_profit.png)

## Limitations and next steps

These are honest caveats, and natural next steps:

1. **Approved-only population (selection bias).** LendingClub data contains only approved loans, so we never see how rejected applicants would have performed. A production model would need reject inference.
2. **Profit is simplified.** "Net profit" is total cash collected minus amount funded. It ignores cost of funds, servicing costs, and the time value of money, and it uses LendingClub's own pricing. Read it as directional.
3. **Calibration drift.** The 2015 vintage defaulted more than 2012–2014 (14.9% vs. 13.3%), so the models under-predict risk in the highest-risk deciles (`reports/fig_calibration.png`). Rank-ordering holds up. Before using these probabilities for pricing or loss forecasting, recalibrate them on recent data (e.g., Platt or isotonic).
4. **"Expanded" ≠ truly alternative data.** These are credit-bureau attributes beyond the score, not cash-flow, rent, or utility data. The same framework (benchmark → challenger with new data → business impact) applies directly to evaluating real alternative data.
5. **Fair lending.** A production model would need disparate-impact testing and adverse-action reason codes. SHAP values provide a starting point for the reason codes.

## Reproduce

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run.sh      # downloads data (~1.6 GB), then prepare -> train -> analyze (~2 min)
```

| File | Purpose |
|---|---|
| `src/prepare_data.py` | Population definition, leakage control, feature engineering |
| `src/train.py` | Out-of-time training and evaluation of all five scorers |
| `src/analyze.py` | Approval and profit policy analysis, SHAP drivers, charts |
| `reports/metrics.json` | Model metrics (all applicants and near-prime) |
| `reports/business_impact.json` | Headline business numbers |

**Data:** LendingClub accepted loans 2007–2018Q4 (public; mirrored on Hugging Face as `codesignal/lending-club-loan-accepted`).
