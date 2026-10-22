"""Realistic trading frictions, day by day and stock by stock.

- Half the bid-ask spread, proportional to the stock's recent volatility (spreads widen when markets are stressed),
  scaled so the median stock in calm years (2015-2019) pays HALF_SPREAD_CALM.
- COMMISSION per side for brokerage, fees and impact.
- Financial transaction taxes on every purchase (opening a long leg or buying back a short), by issuer country.
- Short-selling bans: no new short position in a banned stock (existing positions could be kept).

The Abdi-Ranaldo (2017) spread estimator from daily highs and lows was tried first and rejected: on these liquid
stocks it returned zero for almost half the stock-days.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .data import UNIVERSE

HERE = Path(__file__).parent
TAXES = pd.read_csv(HERE / "taxes.csv", parse_dates=["start", "end"])
BANS = pd.read_csv(HERE / "short_bans.csv", parse_dates=["start", "end"])
COMMISSION = 3e-4
HALF_SPREAD_CALM = 4e-4
FINANCIALS = ("Banks", "Insurance")
LISTING = {"PA": "FR", "DE": "DE", "MI": "IT", "MC": "ES", "AS": "NL", "BR": "BE", "HE": "FI", "VI": "AT", "LS": "PT",
           "IR": "IE"}


def half_spread(px: pd.DataFrame) -> pd.DataFrame:
    vol = px.pct_change(fill_method=None).rolling(21, min_periods=10).std().shift(1)
    calm = vol.loc["2015":"2019"].stack().median()
    return (HALF_SPREAD_CALM * vol / calm).clip(upper=0.01).fillna(HALF_SPREAD_CALM)


def _by_country(px: pd.DataFrame, table: pd.DataFrame, country: pd.Series, value) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for row in table.itertuples():
        cols = [t for t in px.columns if country[t] == row.country and value(row, t) is not None]
        days = (px.index >= row.start) & (pd.isna(row.end) | (px.index <= row.end))
        for t in cols:
            out.loc[days, t] = value(row, t)
    return out


def buy_tax(px: pd.DataFrame) -> pd.DataFrame:
    """Tax rate on purchases. Taxes follow the issuer's home country (Airbus and STMicro are Dutch companies)."""
    return _by_country(px, TAXES, UNIVERSE.country, lambda row, t: row.rate)


def short_ban(px: pd.DataFrame) -> pd.DataFrame:
    """True where a new short position was banned. Bans follow the regulator of the main listing."""
    listing = pd.Series({t: LISTING[t.rsplit(".", 1)[1]] for t in px.columns})
    hit = lambda row, t: 1.0 if row.scope == "all" or UNIVERSE.sector[t] in FINANCIALS else None
    return _by_country(px, BANS, listing, hit).astype(bool)


def frictions(px: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"spread": half_spread(px) + COMMISSION, "tax": buy_tax(px), "ban": short_ban(px)}


def adv(px: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Average daily traded value in euros over the past quarter (primary listing only, so an underestimate)."""
    return (px * volume.reindex_like(px)).rolling(63, min_periods=21).mean()
