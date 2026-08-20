"""
Fama-French 3-factor replication on NSE large-cap portfolio.
Regress portfolio excess return on Mkt-RF, SMB, HML with Newey-West SEs.
Uses flagship prices.csv + ff3.csv (French or synthetic).
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

ROOT = Path(__file__).parent
FF = ROOT / "data" / "ff3.csv"
_cands = [
    Path(__file__).parent.parent.parent / "app-0001-nk-securities-quant-researcher" / "project-backtested-strategy-engine" / "data" / "prices.csv",
    Path(__file__).parent.parent.parent / "app-0001-nk-securities-quant-researcher" / "project" / "data" / "prices.csv",
]
PRICES = next((p for p in _cands if p.exists()), _cands[0])
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

def load():
    # ensure ff3 exists
    if not FF.exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(ROOT / "fetch_french.py")], check=False)
    ff = pd.read_csv(FF, parse_dates=["Date"])
    px = pd.read_csv(PRICES, parse_dates=["Date"])
    print(f"[load] ff {len(ff)} px {len(px)}")
    return ff, px

def portfolio_returns(px):
    # equal-weight portfolio daily return
    px = px.sort_values(["Ticker","Date"])
    px["ret"] = px.groupby("Ticker")["Close"].pct_change()
    daily = px.groupby("Date")["ret"].mean().reset_index()
    daily = daily.dropna()
    return daily

def run():
    ff, px = load()
    port = portfolio_returns(px)
    df = pd.merge(port, ff, on="Date", how="inner")
    df["excess"] = df["ret"] - df["RF"]
    print(f"[merge] {len(df)} overlapping days {df.Date.min()}->{df.Date.max()}")
    # OLS: excess ~ Mkt-RF + SMB + HML
    X = df[["Mkt-RF","SMB","HML"]]
    X = sm.add_constant(X)
    y = df["excess"]
    model = sm.OLS(y, X).fit()
    # Newey-West with 5 lags (hetero + autocorr)
    nw = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags":5})
    print(model.summary())
    print("\n[Newey-West 5 lags]")
    print(nw.summary())

    # Save tables
    def coef_table(res):
        return pd.DataFrame({
            "coef": res.params,
            "std_err": res.bse,
            "t": res.tvalues,
            "p": res.pvalues,
            "ci_low": res.conf_int()[0],
            "ci_high": res.conf_int()[1]
        })

    coef_table(model).to_csv(OUT/"ff3_ols.csv")
    coef_table(nw).to_csv(OUT/"ff3_neweywest.csv")
    with open(OUT/"metrics.json","w") as f:
        json.dump({
            "n": int(len(df)),
            "r2": float(model.rsquared),
            "r2_adj": float(model.rsquared_adj),
            "alpha_ols": float(model.params["const"]),
            "alpha_nw_p": float(nw.pvalues["const"]),
            "betas_ols": model.params.to_dict(),
            "pvalues_nw": nw.pvalues.to_dict()
        }, f, indent=2)

    # Plot: actual vs fitted + residual diagnostics
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    axes[0].scatter(nw.fittedvalues, y, alpha=0.3, s=5)
    axes[0].set_xlabel("Fitted excess return")
    axes[0].set_ylabel("Actual excess return")
    axes[0].set_title(f"FF3 fit R²={model.rsquared:.3f} (n={len(df)})")
    axes[1].plot(df["Date"], nw.resid, alpha=0.6, linewidth=0.5)
    axes[1].set_title("Residuals over time")
    axes[1].set_ylabel("Residual")
    plt.tight_layout()
    plt.savefig(OUT/"ff3_diagnostics.png", dpi=150)
    plt.close()
    print(f"[done] results in {OUT}")

if __name__ == "__main__":
    run()
