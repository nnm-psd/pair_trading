"""Download and clean daily adjusted close prices (Yahoo Finance, dividends and splits included)."""
from pathlib import Path

import pandas as pd

UNIVERSE = pd.read_csv(Path(__file__).with_name("universe.csv")).set_index("ticker")
CORRECTIONS = pd.read_csv(Path(__file__).with_name("corrections.csv"), parse_dates=["date", "end"])
CACHE =Path(__file__).parents[1] / "data" / "prices.csv.gz"
START = "2003-01-01"


def download() -> pd.DataFrame:
    import yfinance as yf

    px = yf.download(list(UNIVERSE.index), start=START, auto_adjust=True, progress=False, threads=True)["Close"]
    missing = sorted(set(UNIVERSE.index) - set(px.columns[px.notna().any()]))
    # A few missing tickers are survivable; a broken feed is not, so fail loudly and keep yesterday's site.
    if len(missing) > 0.1 * len(UNIVERSE):
        raise RuntimeError(f"Yahoo returned no data for {len(missing)} tickers: {missing}")
    CACHE.parent.mkdir(exist_ok=True)
    px.to_csv(CACHE)
    return px


def clean(px: pd.DataFrame) -> pd.DataFrame:
    """Fix the known failure modes of Yahoo's adjusted prices, then fill short gaps (holidays).

    1. Non-positive prices (a broken back-adjustment) become missing.
    2. Days when most markets were shut are dropped.
    3. A >25% jump that comes back within 10% of its start inside 5 days is a bad print: masked.
    4. Confirmed errors in corrections.csv: a range (with `end`) is masked; a single date is neutralised by
       rebasing the history before it.
    Genuine crashes (Fortis 2008, Irish banks) are left alone: removing them would be look-ahead bias.
    """
    px = px.where(px > 0)
    px = px[px.notna().mean(axis=1) >= 0.5]
    prev = px.ffill().shift(1)
    jump = (px / prev - 1).abs() > 0.25
    bad = pd.DataFrame(False, index=px.index, columns=px.columns)
    for k in range(1, 6):
        hit = jump & ((px.shift(-k) / prev - 1).abs() < 0.1)
        for j in range(k):
            bad |= hit.shift(j, fill_value=False)
    px = px.mask(bad)
    for c in CORRECTIONS.itertuples():
        if pd.notna(c.end):
            px.loc[c.date:c.end, c.ticker] = float("nan")
        elif c.ticker in px and c.date in px.index:
            before = px.index < c.date
            px.loc[before, c.ticker] *= px.at[c.date, c.ticker] / px.loc[before, c.ticker].dropna().iloc[-1]
    return px.ffill(limit=5)


def load(cached: bool = False) -> pd.DataFrame:
    px = pd.read_csv(CACHE, index_col=0, parse_dates=True) if cached else download()
    return clean(px[px.columns.intersection(UNIVERSE.index)])
