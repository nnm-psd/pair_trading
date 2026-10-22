"""Download and clean daily adjusted close prices (Yahoo Finance, dividends and splits included)."""
from pathlib import Path

import pandas as pd

UNIVERSE = pd.read_csv(Path(__file__).with_name("universe.csv")).set_index("ticker")
CORRECTIONS = pd.read_csv(Path(__file__).with_name("corrections.csv"), parse_dates=["date", "end"])
CACHE = Path(__file__).parents[1] / "data" / "prices.csv.gz"
START = "2003-01-01"
FILES = {"Adj Close": "prices", "Close": "raw_close", "Dividends": "dividends", "Volume": "volume"}
AUDIT_TOL = 0.02


def _path(name: str) -> Path:
    return CACHE.with_name(f"{name}.csv.gz")


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(_path(name), index_col=0, parse_dates=True)


def download() -> None:
    import yfinance as yf

    raw = yf.download(list(UNIVERSE.index), start=START, auto_adjust=False, actions=True, progress=False, threads=True)
    missing = sorted(set(UNIVERSE.index) - set(raw["Adj Close"].columns[raw["Adj Close"].notna().any()]))
    # A few missing tickers are survivable; a broken feed is not, so fail loudly and keep yesterday's site.
    if len(missing) > 0.1 * len(UNIVERSE):
        raise RuntimeError(f"Yahoo returned no data for {len(missing)} tickers: {missing}")
    CACHE.parent.mkdir(exist_ok=True)
    for field, name in FILES.items():
        raw[field].to_csv(_path(name))


def audit(adj: pd.DataFrame, close: pd.DataFrame, div: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check Yahoo against itself. A day's adjusted return must equal (close + dividend) / previous close - 1 on
    split-adjusted raw prices. Where they disagree by more than AUDIT_TOL the back-adjustment is broken, so the
    raw-based return is used and the price series is rebuilt from returns (anchored to today's adjusted price).
    Returns the rebuilt prices and a mask of the days that were fixed."""
    def ret(x):
        return (x / x.ffill().shift(1) - 1).where(x.notna())

    r_adj = ret(adj.where(adj > 0))
    r_raw = ((close + div.reindex_like(close).fillna(0)) / close.ffill().shift(1) - 1).where(close.notna())
    fixed = ((r_adj - r_raw).abs() > AUDIT_TOL) | (r_adj.isna() & r_raw.notna())
    level = (1 + r_adj.mask(fixed, r_raw).fillna(0)).cumprod().where(close.notna() | adj.notna())
    px = level * (adj.ffill().iloc[-1] / level.ffill().iloc[-1])
    return px, fixed & r_adj.notna() & r_raw.notna()


def clean(px: pd.DataFrame) -> pd.DataFrame:
    """Fix the known failure modes of Yahoo's adjusted prices, then fill short gaps (holidays).

    1. Non-positive prices become missing (audit() already rebuilds them from raw prices where it can).
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


def volume(px: pd.DataFrame) -> pd.DataFrame:
    """Shares traded per day, aligned to the cleaned prices. Read after load(), which writes the file."""
    return _read("volume").reindex(index=px.index, columns=px.columns)


def load(cached: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cleaned prices, and the days the self-audit fixed (for the data-quality report)."""
    if not cached:
        download()
    adj, close, div = (_read(n) for n in ("prices", "raw_close", "dividends"))
    cols = UNIVERSE.index.intersection(adj.columns)
    px, fixed = audit(adj[cols], close.reindex(columns=cols), div.reindex(columns=cols))
    return clean(px), fixed
