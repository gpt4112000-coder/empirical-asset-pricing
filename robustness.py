"""
Robustness checks for the FF3 replication, beyond a single full-sample
Newey-West regression:

1. Rolling-window (252 trading day, ~1yr) betas/alpha -- a coefficient
   that's only "there" in the full-sample fit and unstable rolling through
   time is a classic sign of a spurious full-sample fit, not a real
   loading.
2. Subperiod split (first half vs second half) -- coefficients should be
   broadly consistent across the split if the relationship is real, not
   an artifact of one sub-period.
3. Variance Inflation Factor (VIF) on the three factors -- checks whether
   Mkt-RF/SMB/HML are collinear enough to make individual coefficients
   unstable/hard to interpret, independent of whether the fit is
   "significant."
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ff3_regression import load, portfolio_returns

ROOT = Path(__file__).parent
OUT = ROOT / "results"


def rolling_regression(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    rows = []
    dates = df["Date"].values
    for end in range(window, len(df)):
        seg = df.iloc[end - window:end]
        X = sm.add_constant(seg[["Mkt-RF", "SMB", "HML"]])
        y = seg["excess"]
        try:
            fit = sm.OLS(y, X).fit()
            rows.append({
                "Date": dates[end - 1],
                "alpha": fit.params["const"],
                "beta_mkt": fit.params["Mkt-RF"],
                "beta_smb": fit.params["SMB"],
                "beta_hml": fit.params["HML"],
                "r2": fit.rsquared,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def subperiod_split(df: pd.DataFrame) -> dict:
    mid = len(df) // 2
    halves = {"first_half": df.iloc[:mid], "second_half": df.iloc[mid:]}
    out = {}
    for name, seg in halves.items():
        X = sm.add_constant(seg[["Mkt-RF", "SMB", "HML"]])
        y = seg["excess"]
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        out[name] = {
            "n": int(len(seg)),
            "period": [str(seg["Date"].min()), str(seg["Date"].max())],
            "alpha": float(fit.params["const"]),
            "alpha_p": float(fit.pvalues["const"]),
            "beta_mkt": float(fit.params["Mkt-RF"]),
            "beta_smb": float(fit.params["SMB"]),
            "beta_hml": float(fit.params["HML"]),
            "r2": float(fit.rsquared),
        }
    return out


def compute_vif(df: pd.DataFrame) -> dict:
    X = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]])
    vifs = {X.columns[i]: float(variance_inflation_factor(X.values, i)) for i in range(X.shape[1])}
    return vifs


def main():
    ff, px = load()
    port = portfolio_returns(px)
    df = pd.merge(port, ff, on="Date", how="inner")
    df["excess"] = df["ret"] - df["RF"]
    df = df.sort_values("Date").reset_index(drop=True)

    rolling = rolling_regression(df)
    rolling.to_csv(OUT / "ff3_rolling_betas.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(rolling["Date"], rolling["alpha"], label="alpha", color="black")
    axes[0].axhline(0, color="grey", linewidth=0.8)
    axes[0].set_ylabel("Rolling alpha (252d)")
    axes[0].legend()
    for col, label in [("beta_mkt", "Mkt-RF"), ("beta_smb", "SMB"), ("beta_hml", "HML")]:
        axes[1].plot(rolling["Date"], rolling[col], label=label)
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].set_ylabel("Rolling betas (252d)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT / "ff3_rolling_betas.png", dpi=150)
    plt.close()

    subp = subperiod_split(df)
    vif = compute_vif(df)

    out = {"subperiod_split": subp, "vif": vif,
           "rolling_alpha_std": float(rolling["alpha"].std()) if len(rolling) else None,
           "rolling_beta_mkt_range": [float(rolling["beta_mkt"].min()), float(rolling["beta_mkt"].max())] if len(rolling) else None}
    with open(OUT / "ff3_robustness.json", "w") as f:
        json.dump(out, f, indent=2)

    print("=== VIF (factor multicollinearity) ===")
    print(json.dumps(vif, indent=2))
    print("\n=== Subperiod split ===")
    print(json.dumps(subp, indent=2))
    print(f"\n[done] wrote {OUT / 'ff3_robustness.json'} and {OUT / 'ff3_rolling_betas.png'}")


if __name__ == "__main__":
    main()
