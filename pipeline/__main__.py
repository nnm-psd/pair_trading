"""Run the full study and write site/data/results.json.

    uv run python -m pipeline            # download fresh prices, rates and factors
    uv run python -m pipeline --cached   # reuse the files in data/
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import costs, data, factors, stats, strategy
from .strategy import BASE, ENTRIES, METHODS

OUT = Path(__file__).parents[1] / "site" / "data" / "results.json"
BASE_COST = None  # the realistic cost model in costs.py; COSTS below are flat alternatives
SCREEN = 0.3  # survivorship check: drop the least-traded 30% of stocks at each formation date
CAPACITY_ADV = 0.1  # a trade may take at most this share of a stock's daily volume
COSTS = (0, 5, 10, 20, 30)


def r4(x):
    if isinstance(x, str):
        return x
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def rd(d: dict) -> dict:
    return {k: rd(v) if isinstance(v, dict) else [r4(x) for x in v] if isinstance(v, list) else r4(v) for k, v in d.items()}


def weekly(s: pd.Series) -> list:
    return [r4(v) for v in s.resample("W-FRI").last().dropna()]


LIVE = Path(__file__).parents[1] / "live" / "track.csv"


def record_live(nets: dict, rf: pd.Series) -> pd.DataFrame:
    """Append the headline variants' new daily returns to live/track.csv and return the whole record.

    Only the scheduled CI job writes (CI=true), so a local run can never add to the record. Rows are never
    rewritten: the file is what was published on the day. The first run records only the latest day; a run after
    a gap fills the missing days, and `recorded` shows when each row was written."""
    track = pd.read_csv(LIVE, parse_dates=["date"], dtype={"version": str}) if LIVE.exists() else pd.DataFrame(
        columns=["date", "method", "excess", "rf", "version", "recorded"])
    if os.environ.get("CI") == "true":
        now = datetime.now(timezone.utc).isoformat(timespec="minutes")
        new = []
        for m, r in nets.items():
            done = track[(track.method == m) & (track.version == strategy.MODEL_VERSION)]
            last = done.date.max() if len(done) else r.index[-2]
            new += [{"date": d, "method": m, "excess": round(v, 8), "rf": round(rf[d], 8),
                     "version": strategy.MODEL_VERSION, "recorded": now} for d, v in r[r.index > last].items()]
        if new:
            track = pd.concat([track, pd.DataFrame(new)], ignore_index=True)
            LIVE.parent.mkdir(exist_ok=True)
            track.assign(date=pd.to_datetime(track.date).dt.date).to_csv(LIVE, index=False)
    return track[track.version == strategy.MODEL_VERSION]


def live_summary(track: pd.DataFrame, nets: dict) -> dict:
    """Live cumulative returns next to what the backtest would expect over the same number of days."""
    out = {"version": strategy.MODEL_VERSION, "start": None, "methods": {}}
    if track.empty:
        return out
    out["start"] = pd.to_datetime(track.date).min().date().isoformat()
    for m, g in track.groupby("method"):
        g = g.sort_values("date")
        r, bt = g.excess.astype(float), nets[m]
        n = len(r)
        out["methods"][m] = {
            "dates": [pd.Timestamp(d).date().isoformat() for d in g.date], "cum": [r4(v) for v in (1 + r).cumprod() - 1],
            "days": n, "total": r4((1 + r).prod() - 1), "sharpe": r4(stats.sharpe(r)) if n >= 60 else None,
            # backtest expectation for n days: mean ± 2 standard deviations of the sum of n daily returns
            "expected": r4(bt.mean() * n), "band": r4(2 * bt.std() * np.sqrt(n)),
        }
    return out


def capacity(t: pd.DataFrame, adv: pd.DataFrame) -> dict:
    """Largest portfolio (EUR) at which 95% of the past three years' trades stay within CAPACITY_ADV of each stock's
    average daily traded value. Each pair gets 1/(pairs x 6 overlapping cohorts) of the capital, split by beta."""
    t = t[t.open >= t.open.max() - pd.DateOffset(years=3)]
    share = 1 / (BASE[1] * strategy.TRADE / 21)
    leg_a = 2 / (1 + t.beta) * share
    leg_b = 2 * t.beta / (1 + t.beta) * share
    adv_a = np.array([adv.at[d, a] for d, a in zip(t.open, t.a)])
    adv_b = np.array([adv.at[d, b] for d, b in zip(t.open, t.b)])
    limit = np.minimum(CAPACITY_ADV * adv_a / leg_a, CAPACITY_ADV * adv_b / leg_b)
    return {"aum": r4(np.nanpercentile(limit, 5)), "median_trade": r4(np.nanmedian(limit))}


def survivorship(px, sectors, fr, adv, nets, trades, rf, mkt) -> dict:
    """Two checks on the bias from using today's stock list.
    1. Rerun the headline variants with only the stocks that were among the most traded at each formation date,
       which drops companies that are in the list only because they grew later.
    2. Measure what trades cost when one stock collapsed (fell 40%+ while held): the kind of trade that firms
       missing from the data (Dexia, Wirecard, Banco Popular...) would have produced."""
    eligible = (adv.rank(axis=1, pct=True) > SCREEN).fillna(False)
    screened, *_ = strategy.run(px, sectors, fr | {"eligible": eligible}, grid=False)
    p = px.ffill()
    out = {"screen": SCREEN, "methods": {}}
    for m in METHODS:
        r2 = stats.net(screened[(m, *BASE)], BASE_COST)
        t = trades[trades.method == m]
        move = np.minimum([p.at[c, a] / p.at[o, a] - 1 for o, c, a in zip(t.open, t.close, t.a)],
                          [p.at[c, b] / p.at[o, b] - 1 for o, c, b in zip(t.open, t.close, t.b)])
        hit = t[move <= -0.4]
        out["methods"][m] = {
            "sharpe_all": r4(stats.sharpe(nets[m])), "sharpe_screened": r4(stats.sharpe(r2)),
            "excess_all": r4(nets[m].mean() * 252), "excess_screened": r4(r2.mean() * 252),
            "collapses": len(hit), "collapse_share": r4(len(hit) / len(t)), "collapse_mean": r4(hit.pnl.mean()),
            "collapse_loss_share": r4(hit.pnl.clip(upper=0).sum() / t.pnl.clip(upper=0).sum()),
        }
    return out


def main() -> None:
    t0 = time.time()
    cached = "--cached" in sys.argv
    px, audit_fixed = data.load(cached=cached)
    sectors = data.UNIVERSE.sector
    fr = costs.frictions(px)
    out, trades, live, n_selected = strategy.run(px, sectors, fr)
    print(f"backtest done in {time.time() - t0:.0f}s ({len(out)} variants)")

    adv = costs.adv(px, data.volume(px))
    base = {m: out[(m, *BASE)] for m in METHODS}
    idx = base[METHODS[0]].index
    rf = factors.risk_free(idx, cached)
    mkt = px.pct_change(fill_method=None).mean(axis=1).reindex(idx).fillna(0) - rf  # equal-weight universe, excess
    nets = {m: stats.net(base[m], BASE_COST) for m in METHODS}
    grid = pd.DataFrame({k: stats.net(v, BASE_COST) for k, v in out.items()})
    trial_srs = list(grid.mean() / grid.std())

    F = factors.french(cached).join(factors.reversal(px, sectors), how="inner")
    F.insert(0, "MKT", mkt)
    boot = stats.bootstrap(pd.DataFrame(nets).assign(market=mkt))

    methods = {}
    for m in METHODS:
        r, g = nets[m], base[m].pnl
        t = trades[trades.method == m]
        cols = [k for k in grid.columns if k[0] == m]
        methods[m] = {
            "net": rd(stats.summary(r, mkt, rf) | {"dsr": stats.dsr(r, trial_srs)}),
            "gross": rd(stats.summary(g, mkt, rf)),
            "ci": rd({"sharpe": boot["sharpe_ci"][m], "excess": boot["excess_ci"][m]}),
            "factors": rd(stats.factor_regression(r, F)),
            "factors_mkt_only": rd(stats.factor_regression(r, F[["MKT"]])),
            "halves": rd(stats.halves(r, rf, mkt)),
            "grid": {"sharpe": sorted(r4(stats.sharpe(grid[k])) for k in cols), "pbo": r4(stats.pbo(grid[cols]))},
            "trades": {
                "count": len(t), "per_year": r4(len(t) / (len(idx) / 252)), "win_rate": r4((t.pnl > 0).mean()),
                "mean_pnl": r4(t.pnl.mean()), "median_days": r4(t.days.median()), "converged": r4(t.converged.mean()),
                "pairs_per_cohort": r4(n_selected[m]),
            },
            "turnover": r4(base[m].traded.mean() * 252),
            "cost_breakdown": rd({"gross": base[m].pnl.mean() * 252, "spread": base[m].spread.mean() * 252,
                                  "tax": base[m].tax.mean() * 252, "borrow": stats.BORROW * base[m].short.mean(),
                                  "cost_per_side": base[m].spread.sum() / base[m].traded.sum()}),
            "capacity": capacity(t, adv),
            "equity": weekly((1 + r + rf).cumprod()),
            "equity_gross": weekly((1 + g + rf).cumprod()),
            "drawdown": weekly(stats.drawdown(r + rf)),
            "rolling_sharpe": [r4(v) for v in (r.rolling(252).mean() / r.rolling(252).std() * np.sqrt(252))
                               .resample("ME").last()],
            "yearly": {str(y): r4(v) for y, v in ((1 + r + rf).groupby(r.index.year).prod() - 1).items()},
            "costs": [{"bps": c, "sharpe": r4(stats.sharpe(stats.net(base[m], c))),
                       "cagr": r4(stats.summary(stats.net(base[m], c), mkt, rf)["cagr"])} for c in COSTS],
            "entries": [{"entry": e, "sharpe": [r4(stats.sharpe(stats.net(out[(m, BASE[0], BASE[1], e)], c)))
                                                for c in COSTS]} for e in ENTRIES],
        }

    last = px.index[-1]
    names = data.UNIVERSE.name
    # The same pair can be open in several overlapping cohorts: show it once, from the newest cohort.
    held = {}
    for p in live:
        held.setdefault((p["method"], p["a"], p["b"]), []).append(p)
    newest = [max(ps, key=lambda p: p["cohort"]) | {"cohorts": len(ps)} for ps in held.values()]
    positions = []
    for p in sorted(newest, key=lambda p: -abs(p["z"].iloc[-1])):
        z = p["z"]
        positions.append({
            "method": p["method"], "a": p["a"], "b": p["b"], "a_name": names[p["a"]], "b_name": names[p["b"]],
            "sector": sectors[p["a"]], "cohort": p["cohort"].date().isoformat(),
            "opened": p["opened"].date().isoformat(), "side": p["side"], "z": r4(z.iloc[-1]),
            "pnl": r4(p["pnl"]), "beta": r4(p["beta"]), "cohorts": p["cohorts"],
            "z_dates": [d.date().isoformat() for d in z.index], "z_hist": [r4(v) for v in z],
        })

    mkt_total = mkt + rf
    weeks = (1 + mkt_total).cumprod().resample("W-FRI").last().dropna()
    result = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "data_end": last.date().isoformat(),
        "start": idx[0].date().isoformat(),
        "params": {"formation_days": BASE[0], "trading_days": strategy.TRADE, "pairs": BASE[1], "entry": BASE[2],
                   "exit": strategy.EXIT, "commission": costs.COMMISSION, "half_spread_calm": costs.HALF_SPREAD_CALM,
                   "capacity_adv": CAPACITY_ADV, "borrow": stats.BORROW, "p_max": strategy.P_MAX,
                   "kalman_delta": strategy.KALMAN_DELTA, "costs": list(COSTS), "entries": list(ENTRIES),
                   "forms": list(strategy.FORMS), "ns": list(strategy.NS), "trials": len(trial_srs),
                   "block": stats.BLOCK, "factor_end": F.index[-1].date().isoformat(),
                   "rf": r4(rf.mean() * 252)},
        "universe": [{"ticker": t, "name": names[t], "sector": sectors[t], "country": data.UNIVERSE.country[t]}
                     for t in px.columns],
        "taxes": [{"country": t.country, "start": t.start.date().isoformat(),
                   "end": None if pd.isna(t.end) else t.end.date().isoformat(), "rate": t.rate, "note": t.note}
                  for t in costs.TAXES.itertuples()],
        "bans": [{"country": b.country, "scope": b.scope, "start": b.start.date().isoformat(),
                  "end": b.end.date().isoformat(), "note": b.note} for b in costs.BANS.itertuples()],
        "corrections": [{"ticker": c.ticker, "name": names[c.ticker], "date": c.date.date().isoformat(),
                         "end": None if pd.isna(c.end) else c.end.date().isoformat(), "reason": c.reason}
                        for c in data.CORRECTIONS.itertuples()],
        "candidate_pairs": len(strategy.candidate_pairs(sectors[sectors.index.isin(px.columns)])),
        "cohorts": len(strategy.cohort_starts(px.index)),
        "dates": [d.date().isoformat() for d in weeks.index],
        "rolling_dates": [d.date().isoformat() for d in nets[METHODS[0]].resample("ME").last().index],
        "benchmark": {"equity": [r4(v) for v in weeks], "drawdown": weekly(stats.drawdown(mkt_total)),
                      "summary": rd(stats.summary(mkt, mkt, rf)),
                      "ci": rd({"sharpe": boot["sharpe_ci"]["market"], "excess": boot["excess_ci"]["market"]}),
                      "yearly": {str(y): r4(v) for y, v in ((1 + mkt_total).groupby(mkt.index.year).prod() - 1).items()}},
        "tests": [rd(d) | {"a": d["a"], "b": d["b"]} for d in boot["diffs"]],
        "pbo_all": r4(stats.pbo(grid)),
        "methods": methods,
        "survivorship": survivorship(px, sectors, fr, adv, nets, trades, rf, mkt),
        "data_quality": {"tolerance": data.AUDIT_TOL, "fixed": [
            {"ticker": t, "name": names[t], "date": d.date().isoformat()} for d, t in audit_fixed.stack()[lambda x: x].index]},
        "positions": positions,
        "live": live_summary(record_live(nets, rf), nets),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB) in {time.time() - t0:.0f}s")
    for m in METHODS:
        n, f, h = methods[m]["net"], methods[m]["factors"], methods[m]["halves"]
        print(f"{m:14s} Sharpe {n['sharpe']:.2f} CI {methods[m]['ci']['sharpe']}  DSR {n['dsr']:.2f}  "
              f"alpha {f['alpha']:.2%} (t {f['alpha_t']:.2f}) REV beta {f['betas']['REV']:.2f} (t {f['t']['REV']:.1f})  "
              f"halves {h['first']['sharpe']:.2f}/{h['second']['sharpe']:.2f} p={h['change_p']:.2f}  "
              f"PBO {methods[m]['grid']['pbo']:.2f}")
    print("tests", [(d["a"], d["b"], d["diff"], d["p"]) for d in result["tests"]], "PBO all", result["pbo_all"])


if __name__ == "__main__":
    main()
