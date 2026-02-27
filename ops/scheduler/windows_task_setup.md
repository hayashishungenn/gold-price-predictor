# Windows Task Scheduler Setup

Suggested trigger: daily at 08:30 China time.

Command:
`python src/run_pipeline.py --asof YYYY-MM-DD --train`

Recommended policy:
- Daily run: `--no-train` for fast update
- Weekly run: full `--train` retraining
