from __future__ import annotations

import textwrap
from pathlib import Path


REQUIRED_DIRS = [
    "config",
    "data/raw",
    "data/processed",
    "models/baseline",
    "models/main",
    "reports/daily",
    "reports/backtest",
    "reports/monitoring",
    "src/data_adapters",
    "src/calendar",
    "src/features",
    "src/modeling",
    "src/reporting",
    "ops/scheduler",
    "docs",
]


DEFAULT_FILES: dict[str, str] = {
    "config/symbols.yaml": """
        version: 1

        primary:
          us:
            name: lbma_gold_usd_pm
            market: US
            symbol: LBMA_GOLD_PM_USD
            source: lbma
            currency: USD
            unit: oz
          cn:
            name: sge_au9999_cny_g
            market: CN
            symbol: SGE_AU9999
            source: sge
            currency: CNY
            unit: gram

        fallback:
          us:
            - name: xauusd_spot
              market: US
              symbol: XAUUSD
              source: xauusd
              currency: USD
              unit: oz
          cn:
            - name: shfe_au_main
              market: CN
              symbol: SHFE_AU
              source: shfe
              currency: CNY
              unit: gram
    """,
    "config/targets.yaml": """
        version: 1
        targets:
          next_1d_return:
            task: regression
            horizon: 1
            label: next trading day return
          direction_prob:
            task: classification
            horizon: 1
            positive_label: up
            label: next trading day up probability
          vol_forecast:
            task: regression
            horizon: 1
            label: next trading day volatility proxy
        output_bundle:
          - point_forecast
          - direction_probability
          - prediction_interval
    """,
    "config/sources.yaml": """
        version: 1
        adapters:
          lbma:
            enabled: true
            priority: 1
            endpoint: https://prices.lbma.org.uk/json/gold_pm.json
          xauusd:
            enabled: true
            priority: 2
            ticker: XAUUSD=X
          sge:
            enabled: true
            priority: 1
            symbol: Au99.99
          shfe:
            enabled: true
            priority: 2
            symbol: AU0
          fx:
            enabled: true
            ticker: USDCNY=X
          macro:
            enabled: true
            tickers:
              dxy: DX-Y.NYB
              us10y: ^TNX
              vix: ^VIX
              spx: ^GSPC
              oil: CL=F
        ingestion:
          years_history: 8
          incremental: true
          save_raw_snapshot: true
          allow_synthetic_fallback: true
    """,
    "docs/modeling_strategy.md": """
        # Modeling Strategy

        Route selected: **Route 2 (decomposition)**.

        CN prediction is decomposed into:
        - US gold return forecast
        - USD/CNY return forecast
        - CN premium change forecast

        Final CN return is composed from the three components, so daily report can attribute CN move to US/FX/Premium.
    """,
    "docs/data_dictionary.md": """
        # Data Dictionary

        ## prices_daily
        - date: trading date
        - market: US/CN
        - symbol: instrument id
        - close/open/high/low/volume: daily OHLCV
        - currency: USD/CNY
        - unit: oz/gram
        - source: adapter name

        ## features_daily
        - date: feature date
        - feature_name: canonical feature key
        - value: numeric value
    """,
    "ops/scheduler/windows_task_setup.md": """
        # Windows Task Scheduler Setup

        Suggested trigger: daily at 08:30 China time.

        Command:
        `python src/run_pipeline.py --asof YYYY-MM-DD --train`

        Recommended policy:
        - Daily run: `--no-train` for fast update
        - Weekly run: full `--train` retraining
    """,
    "ops/scheduler/gold_forecast_task.xml": """
        <?xml version="1.0" encoding="UTF-16"?>
        <Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
          <RegistrationInfo>
            <Description>Daily Gold Forecast Pipeline</Description>
          </RegistrationInfo>
          <Triggers>
            <CalendarTrigger>
              <StartBoundary>2026-02-27T08:30:00</StartBoundary>
              <ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay>
            </CalendarTrigger>
          </Triggers>
          <Principals>
            <Principal id="Author">
              <RunLevel>LeastPrivilege</RunLevel>
            </Principal>
          </Principals>
          <Settings>
            <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
            <StartWhenAvailable>true</StartWhenAvailable>
          </Settings>
          <Actions Context="Author">
            <Exec>
              <Command>python</Command>
              <Arguments>src/run_pipeline.py --train</Arguments>
            </Exec>
          </Actions>
        </Task>
    """,
    "ops/logging.yaml": """
        version: 1
        formatters:
          standard:
            format: "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        handlers:
          console:
            class: logging.StreamHandler
            level: INFO
            formatter: standard
        root:
          level: INFO
          handlers: [console]
    """,
    "src/run_pipeline.py": """
        from pathlib import Path
        import sys

        ROOT = Path(__file__).resolve().parents[1]
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        from run_pipeline import main

        if __name__ == "__main__":
            main()
    """,
    "src/data_adapters/__init__.py": """
        from pipeline_data import ingest_all_data
    """,
    "src/calendar/__init__.py": """
        from pipeline_calendar import build_calendar
    """,
    "src/features/__init__.py": """
        from pipeline_features import build_all_features
    """,
    "src/modeling/__init__.py": """
        from pipeline_modeling import train_and_backtest, infer_from_saved_models
    """,
    "src/reporting/__init__.py": """
        from pipeline_reporting import generate_daily_report
    """,
    "src/reporting/generate_daily_report.py": """
        from pathlib import Path
        import sys

        ROOT = Path(__file__).resolve().parents[2]
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        from pipeline_reporting import generate_daily_report

        __all__ = ["generate_daily_report"]
    """,
}


def bootstrap_layout(project_root: Path) -> None:
    for rel_dir in REQUIRED_DIRS:
        (project_root / rel_dir).mkdir(parents=True, exist_ok=True)

    for rel_file, content in DEFAULT_FILES.items():
        file_path = project_root / rel_file
        if file_path.exists():
            continue
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
