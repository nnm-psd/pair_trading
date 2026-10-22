import numpy as np
import pandas as pd

from pipeline import data, stats, strategy


def test_simulate_executes_one_day_late_and_books_pnl_after():
    pa = np.array([10, 10, 10, 10, 9, 8, 8, 8.0])
    pb = np.full(8, 20.0)
    z = np.array([0, 0, 3, 3, 1, -0.5, 0, 0])  # spread rich on day 2, back through 0 on day 5
    r = strategy.simulate(pa, pb, z, np.ones(8), entry=2.0, final=True)
    assert r.traded[3] == 2.0 and r.traded[:3].sum() == 0  # opened at day 3 close, not day 2
    assert r.traded[6] > 0  # closed at day 6 close (signal on day 5)
    # short 1 unit of a at 10 -> 0.1 shares; a falls 10 -> 8, b flat: +0.2
    assert np.isclose(r.pnl.sum(), 0.2) and r.pnl[:4].sum() == 0
    assert len(r.trades) == 1 and r.trades[0]["converged"]


def test_simulate_has_no_look_ahead():
    rng = np.random.default_rng(1)
    pa, pb = 50 + rng.normal(0, 1, 60).cumsum(), 50 + rng.normal(0, 1, 60).cumsum()
    z = rng.normal(0, 2, 60)
    base = strategy.simulate(pa, pb, z, np.ones(60), 2.0, final=False)
    z2 = z.copy()
    z2[30:] = rng.normal(0, 5, 30)  # change the future signal only
    other = strategy.simulate(pa, pb, z2, np.ones(60), 2.0, final=False)
    assert np.allclose(base.pnl[:32], other.pnl[:32])  # a signal on day 30 can only touch P&L from day 32


def test_force_close_at_end_of_period():
    z = np.array([0, 3, 3, 3, 3.0])
    r = strategy.simulate(np.full(5, 10.0), np.full(5, 10.0), z, np.ones(5), 2.0, final=True)
    assert r.traded[-1] > 0 and not r.trades[-1]["converged"]


def test_engle_granger_finds_cointegration_and_rejects_random_walks():
    rng = np.random.default_rng(2)
    n = 252
    b = np.cumsum(rng.normal(0, 0.03, n)) + 3
    e = np.zeros(n)
    for t in range(1, n):  # AR(1) spread, half-life ~ 7 days
        e[t] = 0.9 * e[t - 1] + rng.normal(0, 0.005)
    a = 0.5 + 0.8 * b + e
    eg = strategy.engle_granger(pd.DataFrame({"A": a, "B": b}), [("A", "B")])
    assert eg["p"][0] < 0.05 and abs(eg["beta"][0] - 0.8) < 0.1 and 3 < eg["half_life"][0] < 15
    # independent random walks: the 5% test should reject about 5% of the time (p-values are calibrated)
    walks = pd.DataFrame(np.cumsum(rng.normal(0, 0.02, (n, 600)), axis=0))
    eg = strategy.engle_granger(walks, [(i, i + 300) for i in range(300)])
    assert 0.01 < (eg["p"] < 0.05).mean() < 0.10


def test_kalman_tracks_hedge_ratio():
    # data drawn from the filter's own model: then β is tracked and the standardised errors have unit variance
    rng = np.random.default_rng(3)
    n, w = 1000, 1e-4
    beta_true = 1.5 + np.cumsum(rng.normal(0, np.sqrt(w), n))
    alpha_true = 0.1 + np.cumsum(rng.normal(0, np.sqrt(w), n))
    x = np.cumsum(rng.normal(0, 0.05, n))
    y = beta_true * x + alpha_true + rng.normal(0, 0.01, n)
    z, beta = strategy.kalman(y, x, (1.0, 0.0), ve=1e-4, delta=w / (1 + w))
    assert abs(beta[-1] - beta_true[-1]) < 0.1 and abs(z[200:].std() - 1) < 0.15


def test_clean_fixes_bad_prints_but_keeps_real_crashes():
    idx = pd.bdate_range("2020-01-01", periods=12)
    px = pd.DataFrame({
        "NEG": [-1, -1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        "SPIKE": [10, 10, 10, 30, 31, 10, 10, 10, 10, 10, 10, 10],  # 2-day bad print
        "CRASH": [10, 10, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # real -80%: must stay
    }, index=idx, dtype=float)
    c = data.clean(px)
    assert c.NEG.iloc[:2].isna().all()
    assert (c.SPIKE == 10).all()
    assert c.CRASH.iloc[3] == 2


def test_stats_on_known_returns():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0004, 0.01, 5000), index=pd.bdate_range("2000-01-03", periods=5000))
    s = stats.summary(r, r * 0 + rng.normal(0, 0.01, 5000), r * 0)
    assert abs(s["sharpe"] - 0.0004 / 0.01 * np.sqrt(252)) < 0.25
    assert s["psr"] > 0.95 and -1 < s["max_dd"] < 0
    assert stats.dsr(r, [0.04, 0.0, 0.01, -0.01]) < stats.psr(r)
    assert stats.net(pd.DataFrame({"pnl": [0.01], "traded": [2.0], "short": [0.0]}), 10).iloc[0] == 0.01 - 0.002


def test_pbo_separates_luck_from_skill():
    rng = np.random.default_rng(4)
    noise = pd.DataFrame(rng.normal(0, 0.01, (2000, 20)))
    assert 0.25 < stats.pbo(noise) < 0.75  # pure noise: the in-sample winner is a coin flip out of sample
    skill = noise.copy()
    skill[0] += 0.003  # one variant with a real edge
    assert stats.pbo(skill) < 0.05


def test_bootstrap_intervals_and_difference_test():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2000-01-03", periods=3000)
    R = pd.DataFrame({"good": rng.normal(0.001, 0.01, 3000), "flat": rng.normal(0, 0.01, 3000)}, index=idx)
    b = stats.bootstrap(R, reps=500)
    lo, hi = b["sharpe_ci"]["good"]
    assert lo < stats.sharpe(R.good) < hi and lo > 0
    assert b["diffs"][0]["p"] < 0.05
    same = stats.bootstrap(R.assign(twin=R.flat + rng.normal(0, 0.001, 3000))[["flat", "twin"]], reps=500)
    assert same["diffs"][0]["p"] > 0.05


def test_factor_regression_recovers_loading():
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2000-01-03", periods=3000)
    F = pd.DataFrame({"REV": rng.normal(0, 0.01, 3000), "MKT": rng.normal(0, 0.01, 3000)}, index=idx)
    r = 0.5 * F.REV + rng.normal(0, 0.002, 3000)
    f = stats.factor_regression(r, F)
    assert abs(f["betas"]["REV"] - 0.5) < 0.05 and abs(f["alpha_t"]) < 3


def test_live_record_is_append_only(tmp_path, monkeypatch):
    from pipeline import __main__ as main

    monkeypatch.setattr(main, "LIVE", tmp_path / "track.csv")
    idx = pd.bdate_range("2026-01-05", periods=10)
    rf = pd.Series(0.0001, index=idx)
    nets = {"kalman": pd.Series(0.001, index=idx[:5])}

    monkeypatch.delenv("CI", raising=False)
    assert main.record_live(nets, rf).empty  # a local run never writes
    monkeypatch.setenv("CI", "true")
    assert list(main.record_live(nets, rf).date) == [idx[4]]  # first run: latest day only

    revised = {"kalman": pd.Series(-0.5, index=idx[:8])}  # history rewritten by a later backtest
    track = main.record_live(revised, rf)
    assert len(track) == 4 and track.excess.iloc[0] == 0.001  # old row untouched, three new days appended
