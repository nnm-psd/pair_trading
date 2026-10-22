# Euro Pairs

An open study of pair trading on euro-area large caps, rebuilt after every trading day and published as a static website.

Three classic methods are compared under one protocol fixed in advance:

| Method | Pair selection | Hedge |
|---|---|---|
| Distance (Gatev et al. 2006) | smallest squared distance between normalised prices | 1:1 in euros |
| Cointegration (Engle–Granger, Vidyamurthy 2004) | significant Engle–Granger test, usable half-life | static OLS β |
| Kalman filter (Chan 2013) | same pairs as cointegration | β and α updated daily |

Monthly cohorts form pairs on 252 days and trade them for 126; signals execute at the next close; 10 bps per leg and side plus a 0.5% borrow fee. Results include Newey–West t-statistics, the deflated Sharpe ratio, cost sensitivity and today's open positions. The full write-up is on the site's Method page.

## Run it

```sh
uv sync
uv run pytest                        # 7 offline tests: execution lag, no look-ahead, EG test calibration, Kalman, cleaning, stats
uv run python -m pipeline            # download prices, run the study (~30 s), write site/data/results.json
uv run python -m pipeline --cached   # rerun on the last download
python -m http.server -d site        # preview at http://localhost:8000
```

## Layout

- `pipeline/universe.csv`: the stocks and their sectors. Edit to change the universe.
- `pipeline/corrections.csv`: hand-confirmed errors in Yahoo's adjusted prices.
- `pipeline/data.py`: download and cleaning. `strategy.py`: selection, signals, simulator, cohort loop. `stats.py`: metrics.
- `site/`: static site (no build step). uPlot and the fonts are self-hosted, so pages make no third-party requests.
- `.github/workflows/update.yml`: weekdays 18:30 UTC, test, run, deploy to GitHub Pages. A failure deploys nothing and keeps the previous site.

## Data

Prices come from Yahoo Finance via `yfinance`. Raw prices are neither committed nor published, only derived results. The universe is today's constituents, so results carry survivorship bias (see the Method page).

## Licence

Code: MIT. Results: free to reuse with credit. Research, not investment advice.
