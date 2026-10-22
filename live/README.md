# Live record

`track.csv` holds the daily result of each method's headline variant, after costs, as published on the day.

- Written only by the scheduled GitHub Actions job (`CI=true`); local runs never add rows.
- Append-only: existing rows are never changed, even when a later backtest would give a different number.
- `excess` is the day's return above cash, `rf` the cash return, `version` the model version
  (`MODEL_VERSION` in `pipeline/strategy.py`), `recorded` when the row was written (UTC).
- A model change bumps the version and starts a new record; older versions stay in the file.

The git history of this file is the audit trail.
