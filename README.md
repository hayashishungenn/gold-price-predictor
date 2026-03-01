# gold-price-predictor

Daily US/CN gold forecast pipeline with data ingestion, feature engineering, modeling, backtesting, monitoring, and report generation.

## Data Sources

- US gold: `LBMA` primary, `yfinance` XAUUSD fallback.
- CN gold: `akshare` SGE primary, `akshare` SHFE fallback.
- USD/CNY: ChinaMoney official central parity first, then `yfinance`, then Frankfurter.

The FX fallback chain is implemented with the same ChinaMoney endpoint used by `xalpha`, but kept as native project code because current `xalpha` releases depend on `pandas<2.0` while this project uses `pandas>=2.2`. The multi-source fallback layout also follows the same general approach shown in `daily_stock_analysis`.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py --asof 2026-02-26 --train
```

First run will auto-create the requested project layout (`config/`, `data/`, `models/`, `reports/`, `src/`, `ops/`, `docs/`) and default config files.

Outputs are written to:

- `data/raw/*`
- `data/processed/*`
- `models/*`
- `reports/*`

## GitHub Actions

Workflow file: `.github/workflows/daily_report.yml`

Recommended schedule:

- Trigger at `18:00` Asia/Shanghai, which is `10:00 UTC`.
- The cron is configured on weekdays, and the workflow additionally checks the China trading calendar at runtime so national holidays are skipped automatically.
- This timing is suitable if you want to make a same-day evening buy decision after the domestic session has closed and the day-level CN data is more likely to be complete.

Required repository secrets for email delivery:

- `EMAIL_SENDER`
- `EMAIL_PASSWORD`
- `EMAIL_RECEIVERS`

Manual run:

- Open the `Daily Gold Report` workflow in GitHub Actions.
- Optionally set `asof` to a specific date.
- Leave `train=true` for a full fresh run, or set `train=false` if you later decide to rely on prebuilt models.
