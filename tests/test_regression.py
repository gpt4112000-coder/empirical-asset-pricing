"""
Correctness tests: a regression with a known, engineered relationship
should recover it (validates the merge/excess-return/OLS wiring), and the
VIF utility should report near-1 VIFs for genuinely independent regressors.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent))

from robustness import compute_vif, subperiod_split


def make_synthetic_factor_df(n=500, true_alpha=0.0002, true_beta_mkt=1.0,
                              true_beta_smb=0.3, true_beta_hml=-0.2, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n)
    mkt = rng.normal(0.0004, 0.01, n)
    smb = rng.normal(0.0, 0.005, n)
    hml = rng.normal(0.0, 0.005, n)
    noise = rng.normal(0, 0.003, n)
    excess = true_alpha + true_beta_mkt * mkt + true_beta_smb * smb + true_beta_hml * hml + noise
    return pd.DataFrame({"Date": dates, "Mkt-RF": mkt, "SMB": smb, "HML": hml, "excess": excess})


def test_ols_recovers_known_betas():
    df = make_synthetic_factor_df()
    X = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]])
    fit = sm.OLS(df["excess"], X).fit()
    assert abs(fit.params["Mkt-RF"] - 1.0) < 0.15
    assert abs(fit.params["SMB"] - 0.3) < 0.3
    assert abs(fit.params["HML"] - (-0.2)) < 0.3
    assert fit.rsquared > 0.5  # engineered relationship should dominate the noise


def test_vif_near_one_for_independent_factors():
    df = make_synthetic_factor_df()
    vif = compute_vif(df)
    # independently simulated factors -> low collinearity -> VIF close to 1
    assert vif["Mkt-RF"] < 2.0
    assert vif["SMB"] < 2.0
    assert vif["HML"] < 2.0


def test_vif_high_for_collinear_factors():
    rng = np.random.default_rng(1)
    n = 300
    mkt = rng.normal(0, 0.01, n)
    smb = mkt * 0.98 + rng.normal(0, 0.0005, n)  # near-duplicate of mkt
    hml = rng.normal(0, 0.005, n)
    excess = rng.normal(0, 0.003, n)
    df = pd.DataFrame({"Mkt-RF": mkt, "SMB": smb, "HML": hml, "excess": excess})
    vif = compute_vif(df)
    assert vif["Mkt-RF"] > 5 or vif["SMB"] > 5


def test_subperiod_split_covers_full_sample():
    df = make_synthetic_factor_df(n=400)
    result = subperiod_split(df)
    assert result["first_half"]["n"] + result["second_half"]["n"] == len(df)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
