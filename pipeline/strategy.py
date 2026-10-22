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
from scipy.special import erfi
from statsmodels.tsa.adfvalues import mackinnonp

from . import copula

MODEL_VERSION = "1.0"  # bump whenever the headline model changes; the live record is kept per version
TRADE = 126
FORMS, NS, ENTRIES = (126, 252, 378), (10, 20, 40), (1.5, 2.0, 2.5)
BASE = (252, 20, 2.0)  # (formation days, pairs, entry z)
FIELDS = ("pnl", "traded", "short", "tax", "spread")
EXIT = 0.0
P_MAX = 0.05  # Engle-Granger p-value cut-off
KALMAN_DELTA = 1e-5  # state noise of the Kalman hedge ratio, on log prices rebased to 0 at formation start
METHODS = ("distance", "cointegration", "kalman", "ou", "copula", "pca")
STOP = 4.0  # stop-loss overlay: exit at |z| >= STOP and stay out of the pair for the rest of the period
OU_COST = 0.006  # round-trip cost in log-spread units for Bertram's optimal threshold (~4 legs x 8 bps + tax)
PCA_FACTORS = (5, 10, 15)  # the PCA grid varies the number of factors where the others vary the number of pairs
PCA_SLOTS = 20  # PCA trades the 20 fastest mean-reverting stocks, each against its own factor hedge
PCA_WINDOW = 60  # days of residuals for the OU fit (Avellaneda & Lee)
HEDGE = "HEDGE"  # the b leg of a PCA position: a factor portfolio, not a stock


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


def bertram(theta: float, sigma: float, c: float) -> float:
    """Bertram (2010): for an OU spread dX = -theta X dt + sigma dW (per day) traded symmetrically, enter at -a and
    exit at +a (and the mirror image), the a that maximises expected return per unit of time after a round-trip
    cost c. The expected trade length is E[T] = (2 pi / theta) erfi(a sqrt(theta) / sigma)."""
    eq = sigma / np.sqrt(2 * theta)
    a = np.linspace(c / 2 + 1e-9, 4 * eq, 400)
    mu = (2 * a - c) * theta / (2 * np.pi * erfi(a * np.sqrt(theta) / sigma))
    return float(a[np.argmax(mu)])


def select_pca(form: pd.DataFrame, k: int, n: int) -> list[dict]:
    """Avellaneda & Lee (2010): k eigenportfolios of the formation-period correlation matrix as factors; each
    stock's residual over the last PCA_WINDOW days is fitted as an OU process, and the n fastest mean-reverting
    stocks (reversion within ~30 trading days) are kept."""
    r = np.diff(form.to_numpy(), axis=0)
    sd = r.std(0)
    w, v = np.linalg.eigh(np.corrcoef(((r - r.mean(0)) / sd).T))
    q = v[:, ::-1][:, :k] / sd[:, None]  # eigenportfolio weights
    f = r[-PCA_WINDOW:] @ q
    x = np.c_[np.ones(PCA_WINDOW), f]
    coef = np.linalg.lstsq(x, r[-PCA_WINDOW:], rcond=None)[0]
    cum = np.cumsum(r[-PCA_WINDOW:] - x @ coef, axis=0)
    x0, x1 = cum[:-1], cum[1:]
    b = ((x0 - x0.mean(0)) * (x1 - x1.mean(0))).sum(0) / ((x0 - x0.mean(0)) ** 2).sum(0)
    a = x1.mean(0) - b * x0.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = -np.log(b) * 252
        eq = np.sqrt((x1 - a - b * x0).var(0) / (1 - b**2))
    ok = np.flatnonzero((b > 0) & (b < 1) & (kappa > 252 / 30))
    best = ok[np.argsort(-kappa[ok])][:n]
    return [{"a": form.columns[i], "b": HEDGE, "q": q, "cols": form.columns, "alpha": coef[0, i], "betas": coef[1:, i],
             "m": a[i] / (1 - b[i]), "sd": eq[i], "score": kappa[i]} for i in best]


def prepare(method: str, pair: dict, form: pd.DataFrame, trade_px: pd.DataFrame, s: int, end: int):
    """Everything simulate() needs for one pair: prices of the two legs, z-score, hedge ratio, frictions, and how
    the grid's entry level maps to this method (entry = e * scale; exit at the opposite threshold if flip)."""
    fr = lambda t: [_FR[k][t].to_numpy()[s:end] for k in ("spread", "tax", "ban")]
    pa = trade_px[pair["a"]].to_numpy()
    if method == "pca":
        r = np.r_[np.zeros((1, len(pair["cols"]))), np.diff(np.log(trade_px[pair["cols"]].to_numpy()), axis=0)]
        hedge = (r @ pair["q"]) @ pair["betas"]  # the stock's factor exposure, day by day
        x = np.cumsum(r[:, list(pair["cols"]).index(pair["a"])] - pair["alpha"] - hedge)
        z = (x - pair["m"]) / pair["sd"]
        sa, ta, ba = fr(pair["a"])
        # the hedge is a basket: it pays the average stock's spread and tax, and is never banned (futures exist)
        sb, tb = (_FR[k].iloc[s:end][pair["cols"]].mean(axis=1).to_numpy() for k in ("spread", "tax"))
        frs = (sa, sb, ta, tb, ba, np.zeros(len(z), bool))
        return pa, np.exp(np.cumsum(hedge)), z, np.ones(len(z)), frs, 1.25 / 2, False
    pb = trade_px[pair["b"]].to_numpy()
    (sa, ta, ba), (sb, tb, bb) = fr(pair["a"]), fr(pair["b"])
    frs = (sa, sb, ta, tb, ba, bb)
    if method == "ou":
        e = form[pair["a"]].to_numpy() - pair["beta"] * form[pair["b"]].to_numpy() - pair["alpha"]
        b = (e[1:] @ e[:-1]) / (e[:-1] @ e[:-1])
        if not 0 < b < 1:
            return pa, pb, np.zeros(len(pa)), np.ones(len(pa)), frs, np.inf, True
        theta = -np.log(b)
        sigma = np.std(e[1:] - b * e[:-1], ddof=1) * np.sqrt(2 * theta / (1 - b**2))
        eq = sigma / np.sqrt(2 * theta)
        z = (np.log(pa) - pair["beta"] * np.log(pb) - pair["alpha"]) / eq
        return pa, pb, z, np.full(len(pa), pair["beta"]), frs, bertram(theta, sigma, OU_COST) / eq / 2, True
    if method == "copula":
        ra, rb = (np.diff(form[pair[k]].to_numpy()) for k in ("a", "b"))
        fa, fb = copula.ecdf(ra), copula.ecdf(rb)
        fam, par = copula.fit(fa(ra), fb(rb))
        u = fa(np.diff(np.log(pa), prepend=np.log(pa[0])))
        v = fb(np.diff(np.log(pb), prepend=np.log(pb[0])))
        h1, h2 = copula.h(fam, par, u, v), copula.h(fam, par, v, u)
        h1[0] = h2[0] = 0.5  # no move on the first day
        z = np.cumsum(h1 - 0.5) - np.cumsum(h2 - 0.5)  # > 0: a has risen more than b's move implies
        return pa, pb, z, np.ones(len(pa)), frs, 0.5, False
    z, beta = signals(method, pair, form, trade_px)
    return pa, pb, z, beta, frs, 1.0, False


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
             fr: tuple | None = None, exit: float = EXIT, stop: float | None = None) -> PairResult:
    """fr: per-day arrays for the two stocks (cost per side a, b; purchase tax a, b; short ban a, b).
    Without it, trading is free and unrestricted (P&L before costs). A long spread closes when z >= -exit and a
    short one when z <= exit (exit = -entry closes at the opposite threshold). stop: see STOP."""
    n = len(z)
    if fr is None:
        fr = (np.zeros(n),) * 4 + (np.zeros(n, bool),) * 2
    cost_a, cost_b, tax_a, tax_b, ban_a, ban_b = fr
    res = PairResult(*(np.zeros(n) for _ in FIELDS))
    want = pos = 0  # desired position decided at the close; actual position held
    sa = sb = 0.0  # shares of a and b
    b_sig = 1.0
    opened = None
    stopped = False

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
            if not stopped and beta[t] > 0 and abs(z[t]) > entry:
                want, b_sig = (-1 if z[t] > 0 else 1), beta[t]
        elif stop is not None and abs(z[t]) >= stop:
            want, stopped = 0, True
        elif (want == 1 and z[t] >= -exit) or (want == -1 and z[t] <= exit):
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


def _cohort(s: int, grid: bool = True, stop: float | None = None):
    """One monthly cohort, every variant. Pair P&L does not depend on how many pairs are held, so the best
    max(NS) pairs are simulated once and smaller portfolios are prefixes of that ranking (PCA instead varies its
    number of factors, so each factor count is its own selection)."""
    px, lp = _PX, np.log(_PX)
    end, final = min(s + TRADE, len(px)), s + TRADE <= len(px)
    trade_px, dates = px.iloc[s:end].ffill(), px.index[s:end]
    out, trades, live, n_selected = {}, [], [], {}
    entries = ENTRIES if grid else (BASE[2],)
    for f in FORMS if grid else (BASE[0],):
        form = lp.iloc[s - f:s]
        ok = form.columns[form.notna().all() & lp.iloc[s].notna() & _FR["eligible"].iloc[s]]
        pairs = [p for p in _PAIRS if p[0] in ok and p[1] in ok]
        coint = select_cointegration(form[ok], pairs, max(NS))
        dist = select_distance(form[ok], pairs, max(NS))
        chosen = {"distance": dist, "cointegration": coint, "kalman": coint, "ou": coint, "copula": dist}
        for m in METHODS:
            if m == "pca":
                ks = list(zip(NS, PCA_FACTORS)) if grid else [(BASE[1], PCA_FACTORS[NS.index(BASE[1])])]
                runs = [(n, select_pca(form[ok], k, PCA_SLOTS)) for n, k in ks]
            else:
                runs = [(None, chosen[m])]
            for n_key, sel in runs:
                n_selected[(m, f)] = min(len(sel), BASE[1])  # pairs actually traded by the headline
                prep = [prepare(m, p, form, trade_px, s, end) for p in sel]
                for e in entries:
                    acc = np.zeros((len(FIELDS), max(NS) + 1, end - s))
                    for i, (pair, (pa, pb, z, beta, frs, scale, flip)) in enumerate(zip(sel, prep)):
                        entry = e * scale
                        r = simulate(pa, pb, z, beta, entry, final, frs, -entry if flip else EXIT, stop)
                        acc[:, i + 1] = [getattr(r, x) for x in FIELDS]
                        if (f, e) != (BASE[0], BASE[2]) or i >= BASE[1] or n_key not in (None, BASE[1]):
                            continue
                        trades += [{"method": m, "a": pair["a"], "b": pair["b"], "cohort": dates[0],
                                    "open": dates[t["open"]], "close": dates[t["close"]], "days": t["close"] - t["open"],
                                    "side": t["side"], "beta": t["beta"], "pnl": t["pnl"], "converged": t["converged"]}
                                   for t in r.trades]
                        if r.open:
                            o = r.open
                            live.append({"method": m, "a": pair["a"], "b": pair["b"], "cohort": dates[0],
                                         "opened": dates[o["opened"]], "side": o["side"], "pnl": o["pnl"],
                                         "beta": o["beta"], "entry": entry, "z": pd.Series(z, index=dates)})
                    cum = acc.cumsum(axis=1)
                    if n_key is not None:
                        out[(m, f, n_key, e)] = cum[:, PCA_SLOTS] / PCA_SLOTS
                        continue
                    for n in NS if grid else (BASE[1],):
                        out[(m, f, n, e)] = cum[:, n] / n
    return s, end, out, trades, live, n_selected


def run(px: pd.DataFrame, sectors: pd.Series, fr: dict, grid: bool = True, stop: float | None = None):
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
        for s, end, out, tr, lv, ns in ex.map(partial(_cohort, grid=grid, stop=stop), cohort_starts(px.index), chunksize=4):
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
