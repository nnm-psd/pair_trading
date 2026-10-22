"""Performance statistics. Daily returns in, annualised figures out (252 trading days).

Strategy returns are excess returns: the capital set aside earns the overnight rate (rf) on top, so total
return = excess + rf, and Sharpe ratios, t-stats and factor alphas are on the excess return.
"""
import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

A = 252
BORROW = 0.005  # yearly fee on the short notional
BLOCK = 63  # bootstrap block length: a quarter, longer than the memory left by six overlapping cohorts
HAC = {"cov_type": "HAC", "cov_kwds": {"maxlags": 10}}


def net(df: pd.DataFrame, cost_bps: float) -> pd.Series:
    """Daily excess return after trading costs (bps of notional traded, per leg and side) and the borrow fee."""
    return df.pnl - cost_bps / 1e4 * df.traded - BORROW / A * df.short


def psr(r: pd.Series, sr0: float = 0.0) -> float:
    """Probabilistic Sharpe ratio (Bailey & López de Prado 2012): P(true daily Sharpe > sr0), skew and fat tails included."""
    sr, g3, g4 = r.mean() / r.std(), r.skew(), r.kurt() + 3
    return float(norm.cdf((sr - sr0) * np.sqrt(len(r) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)))


def dsr(r: pd.Series, trial_srs: list[float]) -> float:
    """Deflated Sharpe ratio (Bailey & López de Prado 2014): the PSR against the best Sharpe expected by luck
    from len(trial_srs) tries. trial_srs are daily Sharpe ratios of every variant that was run."""
    n, g = len(trial_srs), np.euler_gamma
    sr0 = np.std(trial_srs, ddof=1) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * np.e)))
    return psr(r, sr0)


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(A))


def summary(r: pd.Series, mkt: pd.Series, rf: pd.Series) -> dict:
    """r and mkt are excess returns; rf is the daily cash return."""
    total = r + rf.reindex(r.index).fillna(0)
    eq = (1 + total).cumprod()
    dd = eq / eq.cummax() - 1
    cagr = eq.iloc[-1] ** (A / len(r)) - 1
    downside = np.sqrt((r.clip(upper=0) ** 2).mean() * A)
    # Newey-West t-stat of the mean daily return: overlapping cohorts make returns autocorrelated
    t = sm.OLS(r.to_numpy(), np.ones(len(r))).fit(**HAC).tvalues[0]
    m = mkt.reindex(r.index).fillna(0)
    monthly = (1 + total).resample("ME").prod() - 1
    return {
        "cagr": cagr, "excess": r.mean() * A, "vol": r.std() * np.sqrt(A), "sharpe": sharpe(r),
        "sortino": r.mean() * A / downside, "max_dd": dd.min(), "calmar": cagr / -dd.min(), "t_stat": t,
        "skew": r.skew(), "kurtosis": r.kurt(), "beta": np.cov(r, m)[0, 1] / m.var(), "corr": r.corr(m),
        "months_up": (monthly > 0).mean(), "worst_month": monthly.min(), "best_month": monthly.max(), "psr": psr(r),
    }


def drawdown(r: pd.Series) -> pd.Series:
    eq = (1 + r).cumprod()
    return eq / eq.cummax() - 1


def bootstrap(R: pd.DataFrame, reps: int = 2000, seed: int = 0) -> dict:
    """Circular block bootstrap of the columns of R, resampled together so their correlation is kept.
    Returns 95% intervals for each column's Sharpe ratio and yearly excess return, and two-sided p-values
    for every pairwise Sharpe difference (basic percentile method)."""
    x = R.to_numpy()
    t = len(x)
    rng = np.random.default_rng(seed)
    blocks = -(-t // BLOCK)
    srs, means = [], []
    for chunk in np.array_split(np.arange(reps), max(reps // 250, 1)):
        starts = rng.integers(0, t, (len(chunk), blocks))
        idx = ((starts[:, :, None] + np.arange(BLOCK)) % t).reshape(len(chunk), -1)[:, :t]
        s = x[idx]  # reps x t x cols
        mu, sd = s.mean(1), s.std(1, ddof=1)
        srs.append(mu / sd * np.sqrt(A))
        means.append(mu * A)
    srs, means = np.vstack(srs), np.vstack(means)
    cols = list(R.columns)
    out = {"sharpe_ci": {c: np.percentile(srs[:, i], [2.5, 97.5]).tolist() for i, c in enumerate(cols)},
           "excess_ci": {c: np.percentile(means[:, i], [2.5, 97.5]).tolist() for i, c in enumerate(cols)},
           "diffs": []}
    for i, j in itertools.combinations(range(len(cols)), 2):
        d = srs[:, i] - srs[:, j]
        observed = sharpe(R.iloc[:, i]) - sharpe(R.iloc[:, j])
        # centre the bootstrap distribution on zero to get the null distribution of the difference
        p = float(np.mean(np.abs(d - d.mean()) >= abs(observed)))
        out["diffs"].append({"a": cols[i], "b": cols[j], "diff": observed,
                             "ci": np.percentile(d, [2.5, 97.5]).tolist(), "p": p})
    return out


def factor_regression(r: pd.Series, F: pd.DataFrame) -> dict:
    """OLS of excess returns on factors with Newey-West errors. Alpha is annualised."""
    df = F.join(r.rename("r"), how="inner").dropna()
    fit = sm.OLS(df.r, sm.add_constant(df.drop(columns="r"))).fit(**HAC)
    return {"alpha": fit.params["const"] * A, "alpha_t": fit.tvalues["const"], "r2": fit.rsquared, "n": int(fit.nobs),
            "betas": {k: fit.params[k] for k in F.columns}, "t": {k: fit.tvalues[k] for k in F.columns}}


def halves(r: pd.Series, rf: pd.Series, mkt: pd.Series) -> dict:
    """Split the sample exactly in half (a rule fixed in advance, not a date picked from the chart) and test
    whether the mean excess return changed (dummy regression, Newey-West errors)."""
    mid = r.index[len(r) // 2]
    first, second = r[r.index < mid], r[r.index >= mid]
    d = sm.OLS(r.to_numpy(), sm.add_constant((r.index >= mid).astype(float))).fit(**HAC)
    return {"split": mid.date().isoformat(), "change_p": float(d.pvalues[1]),
            "first": summary(first, mkt, rf), "second": summary(second, mkt, rf)}


def pbo(M: pd.DataFrame, s: int = 16) -> float:
    """Probability of backtest overfitting by combinatorially symmetric cross-validation (Bailey, Borwein,
    López de Prado & Zhu 2017). M: daily returns, one column per variant. Splits time into s blocks; for each way
    of choosing half the blocks in-sample, picks the best in-sample Sharpe and records its out-of-sample rank.
    PBO = share of splits where that winner lands in the bottom half out of sample."""
    blocks = np.array_split(M.to_numpy(), s)
    stats = [(b.sum(0), (b**2).sum(0), len(b)) for b in blocks]
    logits = []
    for ins in itertools.combinations(range(s), s // 2):
        outs = [i for i in range(s) if i not in ins]

        def sr(ix):
            s1 = sum(stats[i][0] for i in ix)
            s2 = sum(stats[i][1] for i in ix)
            n = sum(stats[i][2] for i in ix)
            mu = s1 / n
            return mu / np.sqrt(s2 / n - mu**2)

        best = np.argmax(sr(ins))
        oos = sr(outs)
        w = (oos < oos[best]).sum() / (len(oos) + 1) + 1 / (len(oos) + 1)  # relative rank in (0, 1)
        logits.append(np.log(w / (1 - w)))
    return float(np.mean(np.array(logits) <= 0))
