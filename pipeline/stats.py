"""Performance statistics. Daily returns in, annualised figures out (252 trading days)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

A = 252
BORROW = 0.005  # yearly fee on the short notional


def net(df: pd.DataFrame, cost_bps: float) -> pd.Series:
    """Daily return after trading costs (bps of notional traded, per leg and side) and the borrow fee."""
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


def summary(r: pd.Series, mkt: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    cagr = eq.iloc[-1] ** (A / len(r)) - 1
    downside = np.sqrt((r.clip(upper=0) ** 2).mean() * A)
    # Newey-West t-stat of the mean daily return: overlapping cohorts make returns autocorrelated
    t = sm.OLS(r.to_numpy(), np.ones(len(r))).fit(cov_type="HAC", cov_kwds={"maxlags": 10}).tvalues[0]
    m = mkt.reindex(r.index).fillna(0)
    monthly = (1 + r).resample("ME").prod() - 1
    return {
        "cagr": cagr, "vol": r.std() * np.sqrt(A), "sharpe": sharpe(r), "sortino": r.mean() * A / downside,
        "max_dd": dd.min(), "calmar": cagr / -dd.min(), "t_stat": t, "skew": r.skew(), "kurtosis": r.kurt(),
        "beta": np.cov(r, m)[0, 1] / m.var(), "corr": r.corr(m), "months_up": (monthly > 0).mean(),
        "worst_month": monthly.min(), "best_month": monthly.max(), "psr": psr(r),
    }


def drawdown(r: pd.Series) -> pd.Series:
    eq = (1 + r).cumprod()
    return eq / eq.cummax() - 1
