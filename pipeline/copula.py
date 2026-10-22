"""Copula pair signal (Liew & Wu 2013; Rad, Low & Faff 2016).

Daily returns of the two stocks are turned into uniforms with each stock's formation-period empirical CDF. Five
one-parameter families (Student-t has two) are fitted by inverting Kendall's tau, the t's degrees of freedom by
likelihood, and the best is chosen by AIC. In trading, h1 = P(U <= u | V = v) and h2 = P(V <= v | U = u) say how
unusual today's move of each stock is given the other's; their running sums above 0.5 are the mispricing indices.
"""
import numpy as np
from scipy import special, stats
from scipy.integrate import quad

NUS = (3, 5, 10, 30)
EPS = 1e-10


def _frank_tau(th: float) -> float:
    return 1 - 4 / th + 4 / th**2 * quad(lambda t: t / np.expm1(t), 0, th)[0]


_TH = np.geomspace(0.01, 80, 400)
_TAU = np.array([_frank_tau(t) for t in _TH])


def logpdf(fam: str, p: tuple, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, v = np.clip(u, EPS, 1 - EPS), np.clip(v, EPS, 1 - EPS)
    if fam == "gaussian":
        (r,) = p
        x, y = special.ndtri(u), special.ndtri(v)
        return -0.5 * np.log(1 - r**2) - (r**2 * (x**2 + y**2) - 2 * r * x * y) / (2 * (1 - r**2))
    if fam == "student":
        r, nu = p
        x, y = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
        return (special.gammaln((nu + 2) / 2) + special.gammaln(nu / 2) - 2 * special.gammaln((nu + 1) / 2)
                - 0.5 * np.log(1 - r**2) - (nu + 2) / 2 * np.log1p((x**2 - 2 * r * x * y + y**2) / (nu * (1 - r**2)))
                + (nu + 1) / 2 * (np.log1p(x**2 / nu) + np.log1p(y**2 / nu)))
    if fam == "clayton":
        (th,) = p
        return np.log1p(th) - (th + 1) * (np.log(u) + np.log(v)) - (2 + 1 / th) * np.log(u**-th + v**-th - 1)
    if fam == "gumbel":
        (th,) = p
        x, y = -np.log(u), -np.log(v)
        s = x**th + y**th
        a = s ** (1 / th)
        return -a - np.log(u) - np.log(v) + (th - 1) * (np.log(x) + np.log(y)) + (1 / th - 2) * np.log(s) + np.log(a + th - 1)
    (th,) = p  # frank
    d = -np.expm1(-th) - np.expm1(-th * u) * np.expm1(-th * v)
    return np.log(th) + np.log(-np.expm1(-th)) - th * (u + v) - 2 * np.log(np.abs(d))


def h(fam: str, p: tuple, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """P(U <= u | V = v)."""
    u, v = np.clip(u, EPS, 1 - EPS), np.clip(v, EPS, 1 - EPS)
    if fam == "gaussian":
        (r,) = p
        return special.ndtr((special.ndtri(u) - r * special.ndtri(v)) / np.sqrt(1 - r**2))
    if fam == "student":
        r, nu = p
        x, y = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
        return stats.t.cdf((x - r * y) / np.sqrt((nu + y**2) * (1 - r**2) / (nu + 1)), nu + 1)
    if fam == "clayton":
        (th,) = p
        return v ** (-th - 1) * (u**-th + v**-th - 1) ** (-1 / th - 1)
    if fam == "gumbel":
        (th,) = p
        x, y = -np.log(u), -np.log(v)
        s = x**th + y**th
        return np.exp(-(s ** (1 / th))) * s ** (1 / th - 1) * y ** (th - 1) / v
    (th,) = p  # frank
    return np.exp(-th * v) * np.expm1(-th * u) / (np.expm1(-th) + np.expm1(-th * u) * np.expm1(-th * v))


def fit(u: np.ndarray, v: np.ndarray) -> tuple[str, tuple]:
    """Best family by AIC. Pairs are positively dependent, so tau is floored at a small positive value."""
    tau = float(np.clip(stats.kendalltau(u, v)[0], 0.01, 0.95))
    r = np.sin(np.pi * tau / 2)
    cands = [("gaussian", (r,)), ("clayton", (2 * tau / (1 - tau),)), ("gumbel", (1 / (1 - tau),)),
             ("frank", (float(np.interp(tau, _TAU, _TH)),))]
    cands += [("student", (r, nu)) for nu in NUS]
    aic = [2 * len(p) - 2 * logpdf(f, p, u, v).sum() for f, p in cands]
    return cands[int(np.argmin(aic))]


def ecdf(sample: np.ndarray):
    s = np.sort(sample)
    n = len(s)
    return lambda x: np.clip(np.searchsorted(s, x, side="right") / (n + 1), 1 / (n + 1), n / (n + 1))
