"""Benchmarks for attribution: the euro risk-free rate, Ken French's European factors, and a reversal factor.

French's European factors are computed in US dollars. Long-short factors (SMB, HML, RMW, CMA, WML) are close to
currency-neutral, so they are used as they are; his market factor is not, so the market is our own universe in euros.
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA = Path(__file__).parents[1] / "data"
ECB = "https://data-api.ecb.europa.eu/service/data/{}?format=csvdata"
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip"


def _cached(name: str, fetch, cached: bool) -> pd.DataFrame:
    path = DATA / f"{name}.csv"
    if cached and path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = fetch()
    DATA.mkdir(exist_ok=True)
    df.to_csv(path)
    return df


def _ecb(key: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(requests.get(ECB.format(key), timeout=60).text), usecols=["TIME_PERIOD", "OBS_VALUE"])
    return df.set_index(pd.to_datetime(df.TIME_PERIOD)).OBS_VALUE.dropna()


def risk_free(index: pd.DatetimeIndex, cached: bool = False) -> pd.Series:
    """Daily return of cash at the overnight rate: EONIA until Sept 2019, then €STR (ACT/360)."""
    def fetch():
        eonia, estr = _ecb("EON/D.EONIA_TO.RATE"), _ecb("EST/B.EU000A2X2A25.WT")
        return pd.concat([eonia[eonia.index < "2019-10-01"], estr]).rename("rate").to_frame()

    rate = _cached("rates", fetch, cached).rate
    rate = rate.reindex(rate.index.union(index)).ffill().reindex(index)
    days = index.to_series().diff().dt.days.fillna(1).to_numpy()
    return pd.Series(rate.shift(1).bfill().to_numpy() / 100 * days / 360, index=index)


def _french_file(name: str) -> pd.DataFrame:
    raw = zipfile.ZipFile(io.BytesIO(requests.get(FRENCH.format(name), timeout=60).content))
    lines = raw.read(raw.namelist()[0]).decode("latin1").splitlines()
    head = next(i for i, l in enumerate(lines) if l.startswith(","))
    rows = [l.split(",") for l in lines[head + 1:] if l[:8].strip().isdigit() and len(l[:8].strip()) == 8]
    cols = ["date"] + [c.strip() for c in lines[head].split(",")[1:]]
    df = pd.DataFrame(rows, columns=cols)
    df.index = pd.to_datetime(df.pop("date").str.strip(), format="%Y%m%d")
    return df.astype(float).replace(-99.99, np.nan) / 100


def french(cached: bool = False) -> pd.DataFrame:
    def fetch():
        f5 = _french_file("Europe_5_Factors_Daily")
        mom = _french_file("Europe_Mom_Factor_Daily")
        return f5[["SMB", "HML", "RMW", "CMA"]].join(mom.iloc[:, 0].rename("WML"), how="inner")

    return _cached("french", fetch, cached)


def reversal(px: pd.DataFrame, sectors: pd.Series) -> pd.Series:
    """Within-sector short-term reversal: each month buy last month's sector losers, sell the winners
    (weights proportional to the return gap to the sector mean, 1 long and 1 short). Pairs trading is known to
    earn much of its return from this effect, so it is the factor that matters most here."""
    r = px.pct_change(fill_method=None)
    month = px.index.to_period("M")
    starts = np.flatnonzero(np.r_[True, month[1:] != month[:-1]])
    w = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    for s in starts[1:]:
        past = px.iloc[s - 1] / px.iloc[max(s - 22, 0)] - 1
        gap = -(past - past.groupby(sectors.reindex(past.index)).transform("mean"))
        w.iloc[s] = (gap / gap.abs().sum() * 2).fillna(0)
    w = w.ffill().fillna(0)
    return (w.shift(1) * r).sum(axis=1).rename("REV")
