# gold-price-predictor

Daily US/CN gold forecast pipeline with data ingestion, feature engineering, modeling, backtesting, monitoring, and report generation.

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
