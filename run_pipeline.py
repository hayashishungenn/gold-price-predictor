from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline_bootstrap import bootstrap_layout
from pipeline_calendar import build_calendar
from pipeline_data import ingest_all_data
from pipeline_features import build_all_features
from pipeline_io import read_dataframe, read_yaml
from pipeline_modeling import infer_from_saved_models, train_and_backtest
from pipeline_monitoring import generate_monitoring_report
from pipeline_reporting import generate_daily_report


LOGGER = logging.getLogger("gold_forecast")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily US/CN gold forecast pipeline.")
    parser.add_argument("--asof", type=str, default=date.today().isoformat(), help="As-of date, format: YYYY-MM-DD")
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=True, help="Train models before report")
    parser.add_argument("--report-only", action="store_true", help="Skip ingestion/training and render report from saved artifacts")
    return parser.parse_args()


def _asof_ts(asof: str) -> pd.Timestamp:
    return pd.to_datetime(asof).tz_localize(None)


def _load_predictions_from_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"market_results": {}, "decomposition": {"available": False}}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_src_entrypoint(project_root: Path) -> None:
    src_file = project_root / "src" / "run_pipeline.py"
    content = """from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_pipeline import main

if __name__ == "__main__":
    main()
"""
    if not src_file.exists():
        src_file.parent.mkdir(parents=True, exist_ok=True)
        src_file.write_text(content, encoding="utf-8")


def main() -> None:
    _setup_logging()
    args = _parse_args()
    project_root = Path(__file__).resolve().parent
    asof = _asof_ts(args.asof)
    LOGGER.info("Pipeline start | asof=%s | train=%s | report_only=%s", asof.date(), args.train, args.report_only)

    bootstrap_layout(project_root)
    _write_src_entrypoint(project_root)

    symbols_cfg = read_yaml(project_root / "config" / "symbols.yaml")
    sources_cfg = read_yaml(project_root / "config" / "sources.yaml")
    _ = read_yaml(project_root / "config" / "targets.yaml")

    if args.report_only:
        LOGGER.info("Report-only mode: loading processed data and latest predictions.")
        prices_daily = read_dataframe(project_root / "data" / "processed" / "prices_daily.parquet")
        features_daily = read_dataframe(project_root / "data" / "processed" / "features_daily.parquet")
        active_symbols = {
            "us": symbols_cfg.get("primary", {}).get("us", {}),
            "cn": symbols_cfg.get("primary", {}).get("cn", {}),
        }
        data_status = {"us": {}, "cn": {}, "us_latest": None, "cn_latest": None}
        pred_payload = _load_predictions_from_json(project_root / "models" / "main" / "latest_predictions.json")
        market_results = {
            "us": {"prediction": pred_payload.get("markets", {}).get("us", {})},
            "cn": {"prediction": pred_payload.get("markets", {}).get("cn", {})},
        }
        decomposition = pred_payload.get("decomposition", {})
    else:
        prices_daily, base_features_daily, active_symbols, data_status = ingest_all_data(
            project_root=project_root,
            symbols_cfg=symbols_cfg,
            sources_cfg=sources_cfg,
            asof=asof.date(),
        )
        calendar_result = build_calendar(
            project_root=project_root,
            prices_daily=prices_daily,
            us_symbol=active_symbols.get("us", {}).get("symbol"),
            cn_symbol=active_symbols.get("cn", {}).get("symbol"),
        )
        for warning in calendar_result.warnings:
            LOGGER.warning("Calendar warning: %s", warning)

        features_result = build_all_features(
            project_root=project_root,
            prices_daily=prices_daily,
            base_features_daily=base_features_daily,
            active_symbols=active_symbols,
            calendar_df=calendar_result.calendar,
        )
        features_daily = features_result.features_daily

        if args.train:
            modeling = train_and_backtest(
                project_root=project_root,
                prices_daily=prices_daily,
                features_daily=features_daily,
                active_symbols=active_symbols,
                asof=asof,
            )
        else:
            modeling = infer_from_saved_models(
                project_root=project_root,
                prices_daily=prices_daily,
                features_daily=features_daily,
                active_symbols=active_symbols,
                asof=asof,
            )
        market_results = modeling["market_results"]
        decomposition = modeling["decomposition"]

    monitoring_path = generate_monitoring_report(
        project_root=project_root,
        asof=asof,
        prices_daily=prices_daily,
        features_daily=features_daily,
        active_symbols=active_symbols,
        data_status=data_status,
    )
    html_path, md_path = generate_daily_report(
        project_root=project_root,
        asof=asof,
        prices_daily=prices_daily,
        features_daily=features_daily,
        market_results=market_results,
        decomposition=decomposition,
        data_status=data_status,
        monitoring_report_path=monitoring_path,
        active_symbols=active_symbols,
    )

    LOGGER.info("Monitoring report: %s", monitoring_path)
    LOGGER.info("Daily report (html): %s", html_path)
    LOGGER.info("Daily report (md): %s", md_path)


if __name__ == "__main__":
    main()
