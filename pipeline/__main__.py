"""Run the full study and write site/data/results.json.

    uv run python -m pipeline            # download fresh prices
    uv run python -m pipeline --cached   # reuse data/prices.csv.gz
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, stats, strategy
from .strategy import BASE_ENTRY, ENTRIES, METHODS

OUT = Path(__file__).parents[1] / "site" / "data" / "results.json"
BASE_COST = 10  # bps per leg and side: commission + half-spread + impact for liquid euro large caps
COSTS = (0, 5, 10, 20, 30)


def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def weekly(s: pd.Series) -> list:
    return [r4(v) for v in s.resample("W-FRI").last().dropna()]


def main() -> None:
    t0 = time.time()
    px = data.load(cached="--cached" in sys.argv)
    sectors = data.UNIVERSE.sector
    out, trades, live, n_selected = strategy.run(px, sectors)
    print(f"backtest done in {time.time() - t0:.0f}s")

    idx = out[(METHODS[0], BASE_ENTRY)].index
    mkt = px.pct_change(fill_method=None).mean(axis=1).reindex(idx).fillna(0)  # equal-weight universe, total return
    nets = {m: stats.net(out[(m, BASE_ENTRY)], BASE_COST) for m in METHODS}
    trial_srs = [stats.net(out[k], BASE_COST).pipe(lambda r: r.mean() / r.std()) for k in out]

    methods = {}
    for m in METHODS:
        r, g = nets[m], out[(m, BASE_ENTRY)].pnl
        t = trades[trades.method == m]
        net_s = stats.summary(r, mkt) | {"dsr": stats.dsr(r, trial_srs)}
        methods[m] = {
            "net": {k: r4(v) for k, v in net_s.items()},
            "gross": {k: r4(v) for k, v in stats.summary(g, mkt).items()},
            "trades": {
                "count": len(t), "per_year": r4(len(t) / (len(idx) / 252)), "win_rate": r4((t.pnl > 0).mean()),
                "mean_pnl": r4(t.pnl.mean()), "median_days": r4(t.days.median()), "converged": r4(t.converged.mean()),
                "pairs_per_cohort": r4(n_selected[m]),
            },
            "turnover": r4(out[(m, BASE_ENTRY)].traded.mean() * 252),
            "equity": weekly((1 + r).cumprod()),
            "equity_gross": weekly((1 + g).cumprod()),
            "drawdown": weekly(stats.drawdown(r)),
            "rolling_sharpe": [r4(v) for v in (r.rolling(252).mean() / r.rolling(252).std() * np.sqrt(252))
                               .resample("ME").last()],
            "yearly": {str(y): r4(v) for y, v in ((1 + r).groupby(r.index.year).prod() - 1).items()},
            "costs": [{"bps": c, "sharpe": r4(stats.sharpe(stats.net(out[(m, BASE_ENTRY)], c))),
                       "cagr": r4(stats.summary(stats.net(out[(m, BASE_ENTRY)], c), mkt)["cagr"])} for c in COSTS],
            "entries": [{"entry": e, "sharpe": [r4(stats.sharpe(stats.net(out[(m, e)], c))) for c in COSTS]}
                        for e in ENTRIES],
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

    weeks = (1 + mkt).cumprod().resample("W-FRI").last().dropna()
    result = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "data_end": last.date().isoformat(),
        "start": idx[0].date().isoformat(),
        "params": {"formation_days": strategy.FORM, "trading_days": strategy.TRADE, "pairs": strategy.N_PAIRS,
                   "entry": BASE_ENTRY, "exit": strategy.EXIT, "cost_bps": BASE_COST, "borrow": stats.BORROW,
                   "p_max": strategy.P_MAX, "kalman_delta": strategy.KALMAN_DELTA, "costs": list(COSTS),
                   "entries": list(ENTRIES), "trials": len(trial_srs)},
        "universe": [{"ticker": t, "name": names[t], "sector": sectors[t]} for t in px.columns],
        "corrections": [{"ticker": c.ticker, "name": names[c.ticker], "date": c.date.date().isoformat(),
                         "end": None if pd.isna(c.end) else c.end.date().isoformat(), "reason": c.reason}
                        for c in data.CORRECTIONS.itertuples()],
        "candidate_pairs": len(strategy.candidate_pairs(sectors[sectors.index.isin(px.columns)])),
        "cohorts": len(strategy.cohort_starts(px.index)),
        "dates": [d.date().isoformat() for d in weeks.index],
        "rolling_dates": [d.date().isoformat() for d in nets[METHODS[0]].resample("ME").last().index],
        "benchmark": {"equity": [r4(v) for v in weeks], "drawdown": weekly(stats.drawdown(mkt)),
                      "summary": {k: r4(v) for k, v in stats.summary(mkt, mkt).items()},
                      "yearly": {str(y): r4(v) for y, v in ((1 + mkt).groupby(mkt.index.year).prod() - 1).items()}},
        "methods": methods,
        "positions": positions,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB) in {time.time() - t0:.0f}s")
    for m in METHODS:
        n = methods[m]["net"]
        print(f"{m:14s} net Sharpe {n['sharpe']:.2f}  CAGR {n['cagr']:.2%}  maxDD {n['max_dd']:.1%}  DSR {n['dsr']:.2f}"
              f"  gross Sharpe {methods[m]['gross']['sharpe']:.2f}  trades {methods[m]['trades']['count']}")


if __name__ == "__main__":
    main()
