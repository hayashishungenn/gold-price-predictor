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

- Weekdays at `09:17` Asia/Shanghai, which is `01:17 UTC`.
- This timing is chosen for same-day buy decisions: it is after the usual morning RMB central parity update window, and still before the `09:30` A-share continuous session.
- The workflow deliberately avoids running at the top of the hour because GitHub Actions scheduled jobs can be delayed more often around `:00`.

Required repository secrets for email delivery:

- `EMAIL_SENDER`
- `EMAIL_PASSWORD`
- `EMAIL_RECEIVERS`

Manual run:

- Open the `Daily Gold Report` workflow in GitHub Actions.
- Optionally set `asof` to a specific date.
- Leave `train=true` for a full fresh run, or set `train=false` if you later decide to rely on prebuilt models.
