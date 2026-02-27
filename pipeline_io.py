from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


LOGGER = logging.getLogger(__name__)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=False, sort_keys=False)


def write_dataframe(df: pd.DataFrame, parquet_path: Path) -> Path:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as exc:
        fallback = parquet_path.with_suffix(".csv")
        LOGGER.warning("Parquet write failed (%s). Falling back to CSV: %s", exc, fallback)
        df.to_csv(fallback, index=False)
        return fallback


def read_dataframe(parquet_path: Path) -> pd.DataFrame:
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception as exc:
            LOGGER.warning("Parquet read failed (%s): %s", exc, parquet_path)
    csv_path = parquet_path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing dataset: {parquet_path} (or CSV fallback)")


def ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col]).dt.tz_localize(None)
    return out


def to_feature_long(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    id_vars = [date_col]
    value_vars = [c for c in df.columns if c not in id_vars]
    long_df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="feature_name", value_name="value")
    long_df = long_df.dropna(subset=["value"])
    return long_df


def from_feature_long(features_daily: pd.DataFrame) -> pd.DataFrame:
    return (
        features_daily.pivot_table(index="date", columns="feature_name", values="value", aggfunc="last")
        .sort_index()
        .reset_index()
    )
