"""Pair selection (distance, Engle-Granger cointegration) and trading (static or Kalman hedge), one shared simulator.

Protocol, fixed before any result was seen:
- Every month a new cohort forms pairs on the past FORM trading days and trades them for the next TRADE days.
- Pairs are only formed inside a sector. Each method keeps its best N_PAIRS.
- A signal seen at the close of day t is executed at the close of day t+1.
- Open when |z| > entry, close when z crosses EXIT, force-close at the end of the trading period.
- Each pair gets 1 unit of capital: 2 units gross (long + short). Cohort return = pair P&L / N_PAIRS (committed capital).
"""
import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.adfvalues import mackinnonp

FORM, TRADE, N_PAIRS = 252, 126, 20
ENTRIES = (1.5, 2.0, 2.5)  # 2.0 is the headline; the others are a robustness check
BASE_ENTRY = 2.0
EXIT = 0.0
P_MAX = 0.05  # Engle-Granger p-value cut-off
KALMAN_DELTA = 1e-5  # state noise of the Kalman hedge ratio, on log prices rebased to 0 at formation start
METHODS = ("distance", "cointegration", "kalman")


def candidate_pairs(sectors: pd.Series) -> list[tuple[str, str]]:
    return [p for _, g in sectors.groupby(sectors) for p in itertools.combinations(sorted(g.index), 2)]


def select_distance(lp: pd.DataFrame, pairs: list) -> list[dict]:
    """Gatev, Goetzmann & Rouwenhorst (2006): smallest sum of squared differences of normalised prices."""
    norm = np.exp(lp - lp.iloc[0]).to_numpy()
    col = {c: i for i, c in enumerate(lp.columns)}
    ia, ib = (np.array([col[p[k]] for p in pairs]) for k in (0, 1))
    d = norm[:, ia] - norm[:, ib]
    ssd = (d**2).sum(0)
    sd = d.std(0, ddof=1)
    return [{"a": pairs[i][0], "b": pairs[i][1], "sd": sd[i], "score": ssd[i]} for i in np.argsort(ssd)[:N_PAIRS]]


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


def select_cointegration(lp: pd.DataFrame, pairs: list) -> list[dict]:
    """Vidyamurthy (2004): significant Engle-Granger pairs with a usable half-life, most significant first."""
    eg = engle_granger(lp, pairs)
    ok = (eg["p"] < P_MAX) & (eg["beta"] > 0) & (eg["half_life"] >= 1) & (eg["half_life"] <= TRADE)
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(eg["t"][idx])][:N_PAIRS]
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
    traded: np.ndarray  # notional traded per day (costs scale with it)
    short: np.ndarray  # short notional held per day (borrow fee scales with it)
    trades: list = field(default_factory=list)
    open: dict | None = None  # the position still open on the last day of an unfinished period


def simulate(pa: np.ndarray, pb: np.ndarray, z: np.ndarray, beta: np.ndarray, entry: float, final: bool) -> PairResult:
    n = len(z)
    res = PairResult(np.zeros(n), np.zeros(n), np.zeros(n))
    want = pos = 0  # desired position decided at the close; actual position held
    sa = sb = 0.0  # shares of a and b
    b_sig = 1.0
    opened = None
    for t in range(n):
        if t:
            res.pnl[t] = sa * (pa[t] - pa[t - 1]) + sb * (pb[t] - pb[t - 1])
        order = 0 if (final and t == n - 1) else want
        if order != pos:
            if pos:
                res.traded[t] += abs(sa) * pa[t] + abs(sb) * pb[t]
                res.trades.append({"open": opened, "close": t, "side": pos,
                                   "pnl": float(res.pnl[opened + 1:t + 1].sum()), "converged": order == 0 and want == 0})
                sa = sb = 0.0
            if order:
                sa, sb = order * 2 / (1 + b_sig) / pa[t], -order * 2 * b_sig / (1 + b_sig) / pb[t]
                res.traded[t] += 2.0
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
    """Positions of the first trading day of each month that has a full formation window behind it."""
    month = index.to_period("M")
    starts = np.flatnonzero(np.r_[True, month[1:] != month[:-1]])
    return [int(s) for s in starts if s >= FORM]


def run(px: pd.DataFrame, sectors: pd.Series):
    """Run every cohort for every method and entry threshold. Returns daily portfolio series and trade records."""
    lp = np.log(px)
    n = len(px)
    pairs_all = candidate_pairs(sectors[sectors.index.isin(px.columns)])
    keys = [(m, e) for m in METHODS for e in ENTRIES]
    acc = {k: {f: np.zeros(n) for f in ("pnl", "traded", "short")} for k in keys}
    active = np.zeros(n)
    trades, live, n_selected = [], [], {m: [] for m in METHODS}

    for s in cohort_starts(px.index):
        end = min(s + TRADE, n)
        final = s + TRADE <= n
        form = lp.iloc[s - FORM:s]
        ok = form.columns[form.notna().all() & lp.iloc[s].notna()]
        pairs = [p for p in pairs_all if p[0] in ok and p[1] in ok]
        trade_px = px.iloc[s:end].ffill()
        coint = select_cointegration(form[ok], pairs)
        chosen = {"distance": select_distance(form[ok], pairs), "cointegration": coint, "kalman": coint}
        active[s:end] += 1
        for m in METHODS:
            n_selected[m].append(len(chosen[m]))
            for pair in chosen[m]:
                z, beta = signals(m, pair, form, trade_px)
                pa, pb = trade_px[pair["a"]].to_numpy(), trade_px[pair["b"]].to_numpy()
                for e in ENTRIES:
                    r = simulate(pa, pb, z, beta, e, final)
                    for f in ("pnl", "traded", "short"):
                        acc[(m, e)][f][s:end] += getattr(r, f) / N_PAIRS
                    if e != BASE_ENTRY:
                        continue
                    dates = px.index[s:end]
                    trades += [{"method": m, "a": pair["a"], "b": pair["b"], "cohort": dates[0],
                                "open": dates[t["open"]], "close": dates[t["close"]], "days": t["close"] - t["open"],
                                "side": t["side"], "pnl": t["pnl"], "converged": t["converged"]} for t in r.trades]
                    if r.open:
                        o = r.open
                        live.append({"method": m, "a": pair["a"], "b": pair["b"], "cohort": dates[0],
                                     "opened": dates[o["opened"]], "side": o["side"], "pnl": o["pnl"],
                                     "beta": o["beta"], "z": pd.Series(z, index=dates)})

    out = {}
    for k, v in acc.items():
        out[k] = pd.DataFrame({f: v[f] / np.maximum(active, 1) for f in v}, index=px.index)[active > 0]
    return out, pd.DataFrame(trades), live, {m: float(np.mean(v)) for m, v in n_selected.items()}
