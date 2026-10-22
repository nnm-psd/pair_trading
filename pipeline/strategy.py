"""Pair selection (distance, Engle-Granger cointegration) and trading (static or Kalman hedge), one shared simulator.

Protocol, fixed before any result was seen:
- Every month a new cohort forms pairs on the past `form` trading days and trades them for the next TRADE days.
- Pairs are only formed inside a sector. Each method keeps its best `n` pairs.
- A signal seen at the close of day t is executed at the close of day t+1.
- Open when |z| > entry, close when z crosses EXIT, force-close at the end of the trading period.
- Each pair gets 1 unit of capital: 2 units gross (long + short). Cohort return = pair P&L / n (committed capital).

The headline is BASE. The full grid FORMS x NS x ENTRIES per method was fixed with it and is only used to measure
overfitting (deflated Sharpe ratio, probability of backtest overfitting), never to pick the headline.
"""
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pandas as pd
from statsmodels.tsa.adfvalues import mackinnonp

MODEL_VERSION = "1.0"  # bump whenever the headline model changes; the live record is kept per version
TRADE = 126
FORMS, NS, ENTRIES = (126, 252, 378), (10, 20, 40), (1.5, 2.0, 2.5)
BASE = (252, 20, 2.0)  # (formation days, pairs, entry z)
FIELDS = ("pnl", "traded", "short", "tax", "spread")
EXIT = 0.0
P_MAX = 0.05  # Engle-Granger p-value cut-off
KALMAN_DELTA = 1e-5  # state noise of the Kalman hedge ratio, on log prices rebased to 0 at formation start
METHODS = ("distance", "cointegration", "kalman")


def candidate_pairs(sectors: pd.Series) -> list[tuple[str, str]]:
    return [p for _, g in sectors.groupby(sectors) for p in itertools.combinations(sorted(g.index), 2)]


def select_distance(lp: pd.DataFrame, pairs: list, n: int) -> list[dict]:
    """Gatev, Goetzmann & Rouwenhorst (2006): smallest sum of squared differences of normalised prices."""
    norm = np.exp(lp - lp.iloc[0]).to_numpy()
    col = {c: i for i, c in enumerate(lp.columns)}
    ia, ib = (np.array([col[p[k]] for p in pairs]) for k in (0, 1))
    d = norm[:, ia] - norm[:, ib]
    ssd = (d**2).sum(0)
    sd = d.std(0, ddof=1)
    return [{"a": pairs[i][0], "b": pairs[i][1], "sd": sd[i], "score": ssd[i]} for i in np.argsort(ssd)[:n]]


def engle_granger(lp: pd.DataFrame, pairs: list) -> dict[str, np.ndarray]:
    """Vectorised Engle-Granger test of log(a) on log(b) for every pair: OLS hedge, then ADF(1) on the residual."""
    x = lp.to_numpy()
    col = {c: i for i, c in enumerate(lp.columns)}
    ia, ib = (np.array([col[p[k]] for p in pairs]) for k in (0, 1))
    a, b = x[:, ia], x[:, ib]
    am, bm = a - a.mean(0), b - b.mean(0)
    beta = (am * bm).sum(0) / (bm**2).sum(0)
    alpha = a.mean(0) - beta * b.mean(0)
    e = am - beta * bm
    de = np.diff(e, axis=0)
    y, x1, x2 = de[1:], e[1:-1], de[:-1]  # Δe_t = ρ e_{t-1} + γ Δe_{t-1} + u_t
    s11, s22, s12 = (x1 * x1).sum(0), (x2 * x2).sum(0), (x1 * x2).sum(0)
    s1y, s2y = (x1 * y).sum(0), (x2 * y).sum(0)
    det = s11 * s22 - s12**2
    rho = (s22 * s1y - s12 * s2y) / det
    gam = (s11 * s2y - s12 * s1y) / det
    s2 = ((y - rho[None] * x1 - gam[None] * x2) ** 2).sum(0) / (len(y) - 2)
    t = rho / np.sqrt(s2 * s22 / det)
    # The 5% critical value is about -3.34, so only compute p-values where they can matter.
    p = np.ones_like(t)
    for i in np.flatnonzero(t < -3.0):
        p[i] = mackinnonp(t[i], regression="c", N=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        half_life = -np.log(2) / np.log1p(rho)
    return {"beta": beta, "alpha": alpha, "sd": e.std(0, ddof=1), "t": t, "p": p, "half_life": half_life}


def select_cointegration(lp: pd.DataFrame, pairs: list, n: int) -> list[dict]:
    """Vidyamurthy (2004): significant Engle-Granger pairs with a usable half-life, most significant first."""
    eg = engle_granger(lp, pairs)
    ok = (eg["p"] < P_MAX) & (eg["beta"] > 0) & (eg["half_life"] >= 1) & (eg["half_life"] <= TRADE)
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(eg["t"][idx])][:n]
    return [{"a": pairs[i][0], "b": pairs[i][1], **{k: float(v[i]) for k, v in eg.items()}, "score": float(eg["p"][i])}
            for i in idx]


def kalman(y: np.ndarray, x: np.ndarray, theta0: tuple[float, float], ve: float, delta: float = KALMAN_DELTA):
    """Chan (2013): y_t = β_t x_t + α_t + v_t, (β, α) a random walk. Returns the standardised one-step forecast
    error and the hedge ratio known before each observation (no look-ahead)."""
    w = delta / (1 - delta)
    b, a = theta0
    p11 = p12 = p22 = 0.0
    z, beta = np.empty(len(y)), np.empty(len(y))
    for t in range(len(y)):
        r11, r12, r22 = p11 + w, p12, p22 + w
        xt = x[t]
        e = y[t] - (b * xt + a)
        q = xt * xt * r11 + 2 * xt * r12 + r22 + ve
        k1, k2 = (r11 * xt + r12) / q, (r12 * xt + r22) / q
        z[t], beta[t] = e / np.sqrt(q), b
        b, a = b + k1 * e, a + k2 * e
        p11, p12, p22 = r11 - k1 * (xt * r11 + r12), r12 - k1 * (xt * r12 + r22), r22 - k2 * (xt * r12 + r22)
    return z, beta


def signals(method: str, pair: dict, form: pd.DataFrame, trade_px: pd.DataFrame):
    """z-score and hedge ratio (value of the b leg per unit of the a leg) for each day of the trading period."""
    pa, pb = trade_px[pair["a"]].to_numpy(), trade_px[pair["b"]].to_numpy()
    if method == "distance":
        return (pa / pa[0] - pb / pb[0]) / pair["sd"], np.ones(len(pa))
    if method == "cointegration":
        z = (np.log(pa) - pair["beta"] * np.log(pb) - pair["alpha"]) / pair["sd"]
        return z, np.full(len(pa), pair["beta"])
    # kalman: rebase log prices to formation start so the intercept and slope have comparable scales
    la0, lb0 = form[pair["a"]].iloc[0], form[pair["b"]].iloc[0]
    fa, fb = form[pair["a"]].to_numpy() - la0, form[pair["b"]].to_numpy() - lb0
    alpha = fa.mean() - pair["beta"] * fb.mean()
    ve = np.var(fa - pair["beta"] * fb - alpha, ddof=1)
    return kalman(np.log(pa) - la0, np.log(pb) - lb0, (pair["beta"], alpha), ve)


@dataclass
class PairResult:
    pnl: np.ndarray  # gross P&L per day, per unit of capital
    traded: np.ndarray  # notional traded per day (flat per-side costs scale with it)
    short: np.ndarray  # short notional held per day (borrow fee scales with it)
    tax: np.ndarray  # transaction tax paid per day (on purchases)
    spread: np.ndarray  # half-spread and commission paid per day, stock by stock
    trades: list = field(default_factory=list)
    open: dict | None = None  # the position still open on the last day of an unfinished period


def simulate(pa: np.ndarray, pb: np.ndarray, z: np.ndarray, beta: np.ndarray, entry: float, final: bool,
             fr: tuple | None = None) -> PairResult:
    """fr: per-day arrays for the two stocks (cost per side a, b; purchase tax a, b; short ban a, b).
    Without it, trading is free and unrestricted (P&L before costs)."""
    n = len(z)
    if fr is None:
        fr = (np.zeros(n),) * 4 + (np.zeros(n, bool),) * 2
    cost_a, cost_b, tax_a, tax_b, ban_a, ban_b = fr
    res = PairResult(*(np.zeros(n) for _ in FIELDS))
    want = pos = 0  # desired position decided at the close; actual position held
    sa = sb = 0.0  # shares of a and b
    b_sig = 1.0
    opened = None

    def book(t, da, db):  # buy (+) or sell (-) da shares of a and db shares of b at t's close
        va, vb = abs(da) * pa[t], abs(db) * pb[t]
        res.traded[t] += va + vb
        res.spread[t] += va * cost_a[t] + vb * cost_b[t]
        res.tax[t] += (da > 0) * va * tax_a[t] + (db > 0) * vb * tax_b[t]

    for t in range(n):
        if t:
            res.pnl[t] = sa * (pa[t] - pa[t - 1]) + sb * (pb[t] - pb[t - 1])
        order = 0 if (final and t == n - 1) else want
        if order and not pos and (ban_a[t] if order == -1 else ban_b[t]):
            order = 0  # the leg to sell short is under a ban: wait, the signal may still be there when it lifts
        if order != pos:
            if pos:
                book(t, -sa, -sb)
                res.trades.append({"open": opened, "close": t, "side": pos, "beta": float(b_sig),
                                   "pnl": float(res.pnl[opened + 1:t + 1].sum()), "converged": order == 0 and want == 0})
                sa = sb = 0.0
            if order:
                sa, sb = order * 2 / (1 + b_sig) / pa[t], -order * 2 * b_sig / (1 + b_sig) / pb[t]
                book(t, sa, sb)
                opened = t
            pos = order
        res.short[t] = max(-sa * pa[t], 0) + max(-sb * pb[t], 0)
        # decide tomorrow's position from today's close
        if want == 0:
            if beta[t] > 0 and abs(z[t]) > entry:
                want, b_sig = (-1 if z[t] > 0 else 1), beta[t]
        elif (want == 1 and z[t] >= -EXIT) or (want == -1 and z[t] <= EXIT):
            want = 0
    if pos and not final:
        res.open = {"opened": opened, "side": pos, "pnl": float(res.pnl[opened + 1:].sum()), "beta": float(b_sig)}
    return res


def cohort_starts(index: pd.DatetimeIndex) -> list[int]:
    """First trading day of each month with the longest formation window behind it, so every variant shares dates."""
    month = index.to_period("M")
    starts = np.flatnonzero(np.r_[True, month[1:] != month[:-1]])
    return [int(s) for s in starts if s >= max(FORMS)]


def variants() -> list[tuple]:
    return [(m, f, n, e) for m in METHODS for f in FORMS for n in NS for e in ENTRIES]


_PX = _PAIRS = _FR = None


def _init(px: pd.DataFrame, pairs: list, fr: dict) -> None:
    global _PX, _PAIRS, _FR
    _PX, _PAIRS, _FR = px, pairs, fr


def _cohort(s: int, grid: bool = True):
    """One monthly cohort, every variant. Pair P&L does not depend on how many pairs are held, so the best
    max(NS) pairs are simulated once and smaller portfolios are prefixes of that ranking."""
    px, lp = _PX, np.log(_PX)
    end, final = min(s + TRADE, len(px)), s + TRADE <= len(px)
    trade_px, dates = px.iloc[s:end].ffill(), px.index[s:end]
    out, trades, live, n_selected = {}, [], [], {}
    for f in FORMS if grid else (BASE[0],):
        form = lp.iloc[s - f:s]
        ok = form.columns[form.notna().all() & lp.iloc[s].notna() & _FR["eligible"].iloc[s]]
        pairs = [p for p in _PAIRS if p[0] in ok and p[1] in ok]
        coint = select_cointegration(form[ok], pairs, max(NS))
        chosen = {"distance": select_distance(form[ok], pairs, max(NS)), "cointegration": coint, "kalman": coint}
        for m in METHODS:
            n_selected[(m, f)] = len(chosen[m])
            legs = [(trade_px[p["a"]].to_numpy(), trade_px[p["b"]].to_numpy()) for p in chosen[m]]
            frs = [tuple(_FR[k][p[leg]].to_numpy()[s:end] for k in ("spread", "tax", "ban") for leg in ("a", "b"))
                   for p in chosen[m]]
            sig = [signals(m, p, form, trade_px) for p in chosen[m]]
            for e in ENTRIES if grid else (BASE[2],):
                acc = np.zeros((len(FIELDS), max(NS) + 1, end - s))
                for i, (pair, (pa, pb), (z, beta), fr) in enumerate(zip(chosen[m], legs, sig, frs)):
                    r = simulate(pa, pb, z, beta, e, final, fr)
                    acc[:, i + 1] = [getattr(r, f) for f in FIELDS]
                    if (f, e) != (BASE[0], BASE[2]) or i >= BASE[1]:
                        continue
                    trades += [{"method": m, "a": pair["a"], "b": pair["b"], "cohort": dates[0],
                                "open": dates[t["open"]], "close": dates[t["close"]], "days": t["close"] - t["open"],
                                "side": t["side"], "beta": t["beta"], "pnl": t["pnl"], "converged": t["converged"]} for t in r.trades]
                    if r.open:
                        o = r.open
                        live.append({"method": m, "a": pair["a"], "b": pair["b"], "cohort": dates[0],
                                     "opened": dates[o["opened"]], "side": o["side"], "pnl": o["pnl"],
                                     "beta": o["beta"], "z": pd.Series(z, index=dates)})
                cum = acc.cumsum(axis=1)
                for n in NS if grid else (BASE[1],):
                    out[(m, f, n, e)] = cum[:, n] / n
    return s, end, out, trades, live, n_selected


def run(px: pd.DataFrame, sectors: pd.Series, fr: dict, grid: bool = True):
    """Run every cohort for every variant (or only BASE if grid=False), in parallel. fr holds the frictions and an
    optional `eligible` mask (stocks allowed into a cohort on its first day). Returns {variant: daily DataFrame of
    FIELDS}, the trades and still-open positions of the BASE variant, and the mean number of pairs found per cohort."""
    fr = {"eligible": pd.DataFrame(True, index=px.index, columns=px.columns)} | fr
    pairs_all = candidate_pairs(sectors[sectors.index.isin(px.columns)])
    n = len(px)
    acc = {k: np.zeros((len(FIELDS), n)) for k in (variants() if grid else [(m, *BASE) for m in METHODS])}
    active = np.zeros(n)
    trades, live, n_selected = [], [], {}
    with ProcessPoolExecutor(os.cpu_count(), initializer=_init, initargs=(px, pairs_all, fr)) as ex:
        for s, end, out, tr, lv, ns in ex.map(partial(_cohort, grid=grid), cohort_starts(px.index), chunksize=4):
            active[s:end] += 1
            for k, v in out.items():
                acc[k][:, s:end] += v
            trades += tr
            live += lv
            for k, v in ns.items():
                n_selected.setdefault(k, []).append(v)
    keep = active > 0
    frames = {k: pd.DataFrame(dict(zip(FIELDS, v / np.maximum(active, 1))), index=px.index)[keep] for k, v in acc.items()}
    return frames, pd.DataFrame(trades), live, {m: float(np.mean(n_selected[(m, BASE[0])])) for m in METHODS}
