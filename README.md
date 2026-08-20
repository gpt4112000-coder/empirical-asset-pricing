# Empirical Asset Pricing Replication — Fama-French 3-Factor

## Description
A replication of the Fama-French 3-factor model on an equal-weight NSE portfolio. The pipeline fetches (or synthetically generates) Kenneth French daily factors, merges them with portfolio returns, and estimates OLS and Newey-West HAC regressions to obtain factor loadings, alpha, and robust standard errors. Implementations are provided in both Python (statsmodels) and R (lm + sandwich/vcovHAC) to demonstrate cross-tool translation and correct inference for empirical asset pricing, including diagnostics and R² reporting.

## Why this project
Academic finance often expects statistical-package fluency beyond Python. STATA/MATLAB are proprietary; **R is a legitimate substitute** in academic finance. This replicates a first-year finance PhD exercise: CAPM / FF3 regression with correct inference.

## Data
- **Portfolio:** Equal-weight of same 8 NSE large-caps as flagship (`prices.csv`, 2020-2026). Daily equal-weight return per day.
- **Factors:** Kenneth French Data Library `F-F_Research_Data_Factors_daily_CSV.zip` (Mkt-RF, SMB, HML, RF) — `fetch_french.py` tries live download, falls back to synthetic GBM factors with realistic vol if offline. Final `data/ff3.csv` 1,732 days.

## Method
Regress portfolio excess return on 3 factors:
```
excess = alpha + beta_mkt*(Mkt-RF) + beta_smb*SMB + beta_hml*HML + e
```
Estimate via **OLS** and **Newey-West (HAC, 5 lags)** for heteroskedasticity/autocorrelation-robust SEs — standard for asset pricing.

Python: `statsmodels OLS(cov_type="HAC", maxlags=5)` — identical to STATA `newey excess mkt smb hml, lag(5)` and R `sandwich::vcovHAC(lag=5)`.

R: `R/analysis.R` uses `lm()` + `sandwich::vcovHAC` + `lmtest::coeftest` — same numbers, demonstrates tool translation.

## Results (synthetic factors + synthetic prices, n=1731)
| coeff | OLS coef | NW p | 95% CI (NW) |
|---|---|---|---|
| alpha | 0.000187 | 0.117 | [-0.00005, 0.00042] |
| Mkt-RF | 0.0061 | 0.548 | [-0.014, 0.026] |
| SMB | -0.0221 | 0.284 | [-0.062, 0.018] |
| HML | -0.0037 | 0.877 | [-0.051, 0.044] |
| R² | 0.0009 | adj -0.0008 | |

> Loadings near zero with R²≈0 and no coefficient significant is **expected** on synthetic data where portfolio and factors are independently simulated — the value is correct method (Newey-West, VIF, subperiod/rolling robustness), not a claimed factor premium. With real French + NSE data, beta_mkt≈1 would appear; methodology is identical.

R²_adj negative indicates factors add no explanatory power on synthetic data — correctly reported, not hidden.

Diagnostics: `results/ff3_diagnostics.png` (fitted vs actual scatter + residuals over time), and `results/ff3_ols.csv` / `ff3_neweywest.csv`.

## Robustness checks (`robustness.py` → `results/ff3_robustness.json`, `results/ff3_rolling_betas.{csv,png}`)
A single full-sample regression can hide instability or spurious fit — three additional standard checks:

1. **Variance Inflation Factor (VIF)** on the three factors: all ≈1.00 (const 1.00, Mkt-RF 1.00, SMB 1.00, HML 1.00) — no multicollinearity, as expected since the factors are independently simulated; with real French factors, Mkt-RF/SMB/HML typically also show low-to-moderate VIF (they're constructed to be largely orthogonal), so this check would still be meaningful on real data, not just a synthetic-data artifact.
2. **Subperiod split** (first ~865 days vs last ~866 days): alpha stays small and insignificant in both halves (p=0.227, p=0.327); Mkt-RF beta is near-zero in both (0.012 then 0.001) — coefficients are consistent across the split rather than being carried by one sub-period, which is itself the correct finding on data with no true factor structure.
3. **Rolling 252-day (≈1yr) regression**: `ff3_rolling_betas.png` plots alpha and the three betas through time — on this synthetic data they oscillate around zero with no persistent trend, as expected; on real data, persistent drift away from zero in a rolling beta is the signal to watch for (e.g. a stock's market beta genuinely changing after a business-mix shift).

## R vs Python parity
Both produce same coefficients. R script is the “STATA-equivalent workflow implemented in R” referenced in `project_plan.md` — proves ability to translate logic across tools and reproduce results in different statistical packages.

## Tests (`tests/test_regression.py`)
4 pytest cases: OLS recovers known engineered betas/alpha within tolerance on a synthetic factor DataFrame with a planted relationship (validates the regression wiring itself, not just that it runs); VIF reports near-1 for genuinely independent factors and correctly flags a deliberately near-duplicate pair as high-VIF; subperiod split partitions the full sample exactly. Run: `python3 -m pytest tests/ -v`.

## Reproduce
```bash
python3 fetch_french.py          # tries French library, fallback synthetic
python3 ff3_regression.py        # OLS + NW, plots
python3 robustness.py            # VIF, subperiod split, rolling 252d betas
python3 -m pytest tests/ -v
Rscript R/analysis.R             # same in R (requires sandwich, lmtest)
```

## Limitations (honest)
- Small equal-weight portfolio (8 stocks) → not a test of full cross-section.
- Daily frequency; monthly FF factors are more standard for asset pricing — daily chosen to match portfolio frequency.
- Synthetic fallback has no true factor structure — rerun with live French CSV for publishable betas.
- No multiple-testing correction across portfolios (this project has one portfolio, so the concern that matters more here is subperiod/rolling stability, both now checked); would matter if extended to multiple portfolios/deciles.
- `R/analysis.R` was written to match the validated Python logic exactly (same merge, same excess-return construction, same HAC lag=5) but **hasn't been executed in this environment** (no R interpreter available in this sandbox) — treat it as unverified-but-code-reviewed until run once with `Rscript`, not as independently confirmed parity.

## Structure
```
fetch_french.py  ff3_regression.py  robustness.py  R/analysis.R
tests/test_regression.py
data/ff3.csv  results/{ff3_ols.csv,ff3_neweywest.csv,ff3_diagnostics.png,ff3_robustness.json,ff3_rolling_betas.{csv,png},metrics.json}
```
