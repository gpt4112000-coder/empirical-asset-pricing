"""Fetch Fama-French 3-factor data or generate synthetic fallback."""
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path(__file__).parent / "data" / "ff3.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

def try_fetch_french():
    import requests, zipfile, io
    # Kenneth French data library: F-F_Research_Data_Factors_daily_CSV.zip
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    print(f"[fetch] trying {url}")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = [n for n in z.namelist() if n.endswith(".CSV")][0]
    df = pd.read_csv(z.open(name), skiprows=3)
    # clean
    df = df[df.iloc[:,0].astype(str).str.match(r"^\d{8}$", na=False)]
    df.columns = ["Date","Mkt-RF","SMB","HML","RF"]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    # divide by 100 (percent)
    for c in ["Mkt-RF","SMB","HML","RF"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100
    df = df.dropna()
    df = df[(df["Date"] >= "2020-01-01") & (df["Date"] <= "2026-08-20")]
    print(f"[fetch] French data {len(df)} rows {df.Date.min()}->{df.Date.max()}")
    return df

def synthetic():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01","2026-08-20")
    n = len(dates)
    # simulate factors with realistic means
    mkt_rf = rng.normal(0.0003, 0.012, n)
    smb = rng.normal(0.0001, 0.006, n)
    hml = rng.normal(0.00005, 0.005, n)
    rf = np.full(n, 0.00002)
    df = pd.DataFrame({"Date": dates, "Mkt-RF": mkt_rf, "SMB": smb, "HML": hml, "RF": rf})
    print(f"[synthetic] {len(df)} rows")
    return df

if __name__ == "__main__":
    try:
        df = try_fetch_french()
        source = "French library"
    except Exception as e:
        print(f"[warn] fetch failed ({e}), using synthetic")
        df = synthetic()
        source = f"synthetic fallback ({e})"
    df.to_csv(OUT, index=False)
    print(f"[done] Wrote {OUT} source={source}")
