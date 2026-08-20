# Empirical Asset Pricing Replication — Fama-French 3-Factor

**For:** Research Associate Programme, ISB Centre for Analytical Finance — closes SAS/STATA/MATLAB gap with R-equivalent workflow.

## Description
A replication of the Fama-French 3-factor model on an equal-weight NSE portfolio. The pipeline fetches (or synthetically generates) Kenneth French daily factors, merges them with portfolio returns, and estimates OLS and Newey-West HAC regressions to obtain factor loadings, alpha, and robust standard errors. Implementations are provided in both Python (statsmodels) and R (lm + sandwich/vcovHAC) to demonstrate cross-tool translation and correct inference for empirical asset pricing, including diagnostics and R² reporting.

## Why this project
CAF wants statistical-package fluency beyond Python. STATA/MATLAB are proprietary; **R is a legitimate substitute** in academic finance (per `project_plan.md`). This replicates a first-year finance PhD exercise: CAPM / FF3 regression with correct inference.

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
| coeff | OLS coef | OLS p | NW p | 95% CI (NW) |
|---|---|---|---|---|
| alpha | 0.00046 | 0.0003 | 0.0003 | [0.0002, 0.0007] |
| Mkt-RF | -0.0036 | 0.736 | 0.736 | [-0.024, 0.017] |
| SMB | 0.0058 | 0.785 | 0.788 | [-0.037, 0.048] |
| HML | 0.0034 | 0.896 | 0.900 | [-0.049, 0.056] |
| R² | 0.0001 | adj -0.0016 | | |

> Loadings near zero with R²≈0 is **expected** on synthetic data where portfolio and factors are independently simulated — the value is correct method (Newey-West, diagnostics), not a claimed factor premium. With real French + NSE data, beta_mkt≈1 would appear; methodology is identical.

R²_adj negative indicates factors add no explanatory power on synthetic data — correctly reported, not hidden.

Diagnostics: `results/ff3_diagnostics.png` (fitted vs actual scatter + residuals over time), and `results/ff3_ols.csv` / `ff3_neweywest.csv`.

## R vs Python parity
Both produce same coefficients. R script is the “STATA-equivalent workflow implemented in R” referenced in `project_plan.md` — proves ability to translate logic across tools, which CAF interview will test (“could you redo this in STATA/MATLAB live?”).

## Reproduce
```bash
python3 fetch_french.py          # tries French library, fallback synthetic
python3 ff3_regression.py        # OLS + NW, plots
Rscript R/analysis.R             # same in R (requires sandwich, lmtest)
```

## Limitations (honest)
- Small equal-weight portfolio (8 stocks) → not a test of full cross-section.
- Daily frequency; monthly FF factors are more standard for asset pricing — daily chosen to match portfolio frequency.
- Synthetic fallback has no true factor structure — rerun with live French CSV for publishable betas.
- No multiple-testing correction; larger sample + more portfolios needed for publication.

## Structure
```
fetch_french.py  ff3_regression.py  R/analysis.R
data/ff3.csv  results/{ff3_ols.csv,ff3_neweywest.csv,ff3_diagnostics.png,metrics.json}
```
