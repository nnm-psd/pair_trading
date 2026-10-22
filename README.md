# Euro Pairs

An open study of pair trading on euro-area large caps, rebuilt after every trading day and published as a static website.

Six methods share one simulator and one protocol, fixed before any result was seen:

| Method | Pairs | Signal and hedge |
|---|---|---|
| Distance (Gatev et al. 2006) | smallest squared distance of normalised prices | 1:1 in euros |
| Cointegration (Engle–Granger; Vidyamurthy 2004) | significant EG test, usable half-life | static OLS β |
| Kalman filter (Chan 2013) | cointegration pairs | β, α updated daily |
| OU thresholds (Bertram 2010) | cointegration pairs | per-pair optimal entry, exit at the opposite threshold |
| Copula (Liew & Wu 2013; Rad et al. 2016) | distance pairs | best of five copulas by AIC, cumulative mispricing index |
| PCA residuals (Avellaneda & Lee 2010) | 20 fastest mean-reverting stocks | s-score against a daily-rebalanced factor hedge |

Monthly cohorts form on 252 days and trade for 126; signals execute at the next close. Costs are stock by stock:
volatility-scaled half-spread plus 3 bps commission, French/Italian/Spanish transaction taxes and Irish stamp duty
on purchases, 0.5% borrow fee, and no new shorts during regulators' short-selling bans (2008–2020). Capital earns
EONIA/€STR.

The evidence on the site: block-bootstrap intervals and tests between methods, factor attribution (French European
factors plus a within-sector reversal factor), equal-halves test, a pre-registered grid of 168 variants feeding the
deflated Sharpe ratio and the probability of backtest overfitting, a stop-loss overlay, survivorship checks, a
self-audit of Yahoo's price adjustments, capacity, and an append-only live record.

## Run it

```sh
uv sync
uv run pytest                        # 18 offline tests
uv run python -m pipeline            # download prices, rates, factors; run every variant (~4 min on 8 cores)
uv run python -m pipeline --cached   # rerun on the last download
python -m http.server -d site        # preview at http://localhost:8000
```

## Layout

- `pipeline/universe.csv`: stocks, sectors, issuer country (taxes follow the issuer). `corrections.csv`: hand-confirmed
  price errors. `taxes.csv`, `short_bans.csv`: frictions, with sources noted.
- `pipeline/data.py`: download, self-audit, cleaning. `strategy.py`: the six methods, simulator, cohort loop.
  `copula.py`: copula families. `costs.py`: frictions and capacity. `factors.py`: cash rate and factors.
  `stats.py`: statistics and overfitting tests. `__main__.py`: runs everything, writes `site/data/`.
- `live/track.csv`: the live record, written only by the scheduled job, never rewritten. `MODEL_VERSION` in
  `strategy.py` starts a new record when the method changes.
- `site/`: static site, no build step; uPlot and fonts are self-hosted.
- `.github/workflows/update.yml`: weekdays 18:30 UTC: test, run, commit the live record, deploy to GitHub Pages.
  A failure deploys nothing and keeps the previous site.

## Data and limits

Prices come from Yahoo Finance; only derived results are published. The universe is today's companies: firms that
failed or were taken over are missing (Yahoo deletes their history), so results are probably too kind. See the
method page for every assumption.

## Licence

Code: MIT. Results: free to reuse with credit (`CITATION.cff`). Research, not investment advice.
