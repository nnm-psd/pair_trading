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
    day = pd.DataFrame({"pnl": [0.01], "traded": [2.0], "short": [0.0], "tax": [0.003], "spread": [0.0014]})
    assert np.isclose(stats.net(day, 10).iloc[0], 0.01 - 0.002 - 0.003)  # flat cost + tax
    assert np.isclose(stats.net(day).iloc[0], 0.01 - 0.0014 - 0.003)  # realistic cost + tax


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


def test_frictions_charge_tax_on_purchases_and_respect_bans():
    n = 6
    z = np.array([0, 3, 3, 1, -1, 0.0])  # rich spread: short a, long b; converges on day 4
    one = np.ones(n)
    fr = (one * 1e-3, one * 1e-3, one * 0.003, np.zeros(n), np.zeros(n, bool), np.zeros(n, bool))
    r = strategy.simulate(np.full(n, 10.0), np.full(n, 10.0), z, one, 2.0, True, fr)
    assert np.isclose(r.tax[2], 0) and np.isclose(r.tax[5], 0.003)  # a is sold at entry, bought back at exit: taxed
    assert np.isclose(r.spread[2], 2e-3) and np.isclose(r.spread[5], 2e-3)
    banned = (*fr[:4], np.array([0, 0, 1, 1, 0, 0], bool), np.zeros(n, bool))
    r = strategy.simulate(np.full(n, 10.0), np.full(n, 10.0), z, one, 2.0, True, banned)
    assert r.traded[2] == 0 and r.traded[3] == 0  # cannot short a while it is banned


def test_copula_densities_and_conditionals_are_consistent():
    from pipeline import copula

    g = (np.arange(400) + 0.5) / 400  # midpoint grid on (0, 1)
    U, V = np.meshgrid(g, g, indexing="ij")
    for fam, p in [("gaussian", (0.6,)), ("student", (0.6, 5)), ("clayton", (2.0,)), ("gumbel", (2.0,)), ("frank", (5.0,))]:
        c = np.exp(copula.logpdf(fam, p, U, V))
        assert abs(c.mean() - 1) < 0.03, fam  # a density integrates to 1
        # h(u|v) is the integral of the density in u
        integral = np.cumsum(c, axis=0)[200, 100] / 400
        assert abs(copula.h(fam, p, g[200] + 0.5 / 400, g[100]) - integral) < 0.02, fam


def test_copula_fit_recovers_family_strength():
    from pipeline import copula

    rng = np.random.default_rng(7)
    z = rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], 2000)
    u, v = (pd.Series(z[:, i]).rank().to_numpy() / 2001 for i in (0, 1))
    fam, p = copula.fit(u, v)
    assert fam in ("gaussian", "student", "frank")  # symmetric families; Clayton/Gumbel have one-sided tails
    if fam != "frank":
        assert abs(p[0] - 0.7) < 0.08


def test_bertram_threshold_widens_with_cost():
    a0, a1 = strategy.bertram(0.05, 0.01, 0.0), strategy.bertram(0.05, 0.01, 0.01)
    assert 0 < a0 < a1 and a1 > 0.005  # must at least cover half the round-trip cost on each side


def test_exit_at_opposite_threshold_and_stop_loss():
    z = np.array([0, -3, -1, 0.5, 2.5, 0, 0.0])
    r = strategy.simulate(np.full(7, 10.0), np.full(7, 10.0), z, np.ones(7), 2.0, True, exit=-2.0)
    assert r.trades[0]["open"] == 2 and r.trades[0]["close"] == 5  # held through zero until z > +2
    z = np.array([0, 3, 4.5, 5, 3, 3, 0.0])  # diverges further: stop at 4, and no re-entry at 3
    r = strategy.simulate(np.full(7, 10.0), np.full(7, 10.0), z, np.ones(7), 2.0, True, stop=4.0)
    assert len(r.trades) == 1 and r.trades[0]["close"] == 3


def test_pca_selects_the_mean_reverting_stock():
    rng = np.random.default_rng(8)
    t, n = 253, 12
    factor = rng.normal(0, 0.01, t)
    noise = rng.normal(0, 0.01, (t, n)).cumsum(0) * 0.05 + rng.normal(0, 0.004, (t, n))
    e = np.zeros(t)
    for i in range(1, t):  # stock 0's residual reverts fast
        e[i] = 0.6 * e[i - 1] + rng.normal(0, 0.01)
    r = factor[:, None] + noise
    r[:, 0] = factor + np.diff(e, prepend=0)
    lp = pd.DataFrame(np.cumsum(r, axis=0), columns=[f"S{i}" for i in range(n)])
    sel = strategy.select_pca(lp, 1, 3)
    assert sel and sel[0]["a"] == "S0"


def test_pca_daily_signals_have_no_look_ahead():
    rng = np.random.default_rng(9)
    n, t = 10, 400
    lp = pd.DataFrame(np.cumsum(rng.normal(0, 0.01, (t, n)), axis=0), columns=[f"S{i}" for i in range(n)])
    s, end = 300, 360
    sel = strategy.select_pca(lp.iloc[s - 252:s], 2, 3)
    assert sel
    first = [dict(p) for p in sel]
    strategy.pca_signals(first, lp, s, end)
    later = lp.copy()
    later.iloc[s + 30:] += rng.normal(0, 0.05, (t - s - 30, n)).cumsum(0)  # rewrite the future after day 30
    second = [dict(p) for p in sel]
    strategy.pca_signals(second, later, s, end)
    for a, b in zip(first, second):
        assert np.allclose(a["z"][:30], b["z"][:30]) and np.allclose(a["hedge"][:30], b["hedge"][:30])
