from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline_data import OZ_TO_GRAM
from pipeline_io import ensure_datetime, from_feature_long, to_feature_long, write_dataframe


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _single_market_price_features(price_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = price_df.sort_values("date").copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out = pd.DataFrame({"date": df["date"].values})
    out[f"{prefix}_close"] = close.values
    out[f"{prefix}_ret_1d"] = close.pct_change(1).values
    out[f"{prefix}_ret_5d"] = close.pct_change(5).values
    out[f"{prefix}_ret_20d"] = close.pct_change(20).values
    out[f"{prefix}_vol_5d"] = out[f"{prefix}_ret_1d"].rolling(5, min_periods=5).std().values
    out[f"{prefix}_vol_20d"] = out[f"{prefix}_ret_1d"].rolling(20, min_periods=20).std().values
    out[f"{prefix}_ma_5"] = close.rolling(5, min_periods=5).mean().values
    out[f"{prefix}_ma_20"] = close.rolling(20, min_periods=20).mean().values
    out[f"{prefix}_ma_ratio"] = (out[f"{prefix}_ma_5"] / out[f"{prefix}_ma_20"]).values
    out[f"{prefix}_rsi_14"] = _compute_rsi(close, window=14).values
    out[f"{prefix}_atr_14"] = true_range.rolling(14, min_periods=14).mean().values
    return out


def _series_for_symbol(prices_daily: pd.DataFrame, market: str, symbol: str) -> pd.DataFrame:
    df = prices_daily.loc[(prices_daily["market"] == market) & (prices_daily["symbol"] == symbol)].copy()
    if df.empty:
        return df
    return df.sort_values("date").drop_duplicates(subset=["date"], keep="last")


def build_price_features(
    project_root: Path,
    prices_daily: pd.DataFrame,
    active_symbols: dict[str, Any],
) -> pd.DataFrame:
    prices = ensure_datetime(prices_daily, "date")
    us_symbol = active_symbols.get("us", {}).get("symbol")
    cn_symbol = active_symbols.get("cn", {}).get("symbol")

    us_series = _series_for_symbol(prices, "US", us_symbol) if us_symbol else pd.DataFrame()
    cn_series = _series_for_symbol(prices, "CN", cn_symbol) if cn_symbol else pd.DataFrame()

    pieces: list[pd.DataFrame] = []
    if not us_series.empty:
        pieces.append(_single_market_price_features(us_series, prefix="us"))
    if not cn_series.empty:
        pieces.append(_single_market_price_features(cn_series, prefix="cn"))

    if not pieces:
        out = pd.DataFrame(columns=["date", "feature_name", "value"])
        write_dataframe(out, project_root / "data" / "processed" / "features_prices.parquet")
        return out

    wide = pieces[0]
    for p in pieces[1:]:
        wide = wide.merge(p, on="date", how="outer")
    wide = wide.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    out = to_feature_long(wide, date_col="date")
    out = ensure_datetime(out, "date")
    write_dataframe(out, project_root / "data" / "processed" / "features_prices.parquet")
    return out


def build_crossmarket_features(
    project_root: Path,
    prices_daily: pd.DataFrame,
    base_features_daily: pd.DataFrame,
    active_symbols: dict[str, Any],
) -> pd.DataFrame:
    prices = ensure_datetime(prices_daily, "date")
    feat_wide = from_feature_long(ensure_datetime(base_features_daily, "date")) if not base_features_daily.empty else pd.DataFrame()

    us_symbol = active_symbols.get("us", {}).get("symbol")
    cn_symbol = active_symbols.get("cn", {}).get("symbol")
    if not us_symbol or not cn_symbol:
        out = pd.DataFrame(columns=["date", "feature_name", "value"])
        write_dataframe(out, project_root / "data" / "processed" / "features_crossmarket.parquet")
        return out

    us_df = _series_for_symbol(prices, "US", us_symbol)[["date", "close", "unit"]].rename(columns={"close": "us_gold_usd"})
    cn_df = _series_for_symbol(prices, "CN", cn_symbol)[["date", "close", "unit"]].rename(columns={"close": "cn_gold_cny"})
    if us_df.empty or cn_df.empty:
        out = pd.DataFrame(columns=["date", "feature_name", "value"])
        write_dataframe(out, project_root / "data" / "processed" / "features_crossmarket.parquet")
        return out

    fx_cols = ["date", "usdcny_close"] if ("usdcny_close" in feat_wide.columns if not feat_wide.empty else False) else ["date"]
    fx_df = feat_wide[fx_cols].copy() if not feat_wide.empty else pd.DataFrame(columns=["date", "usdcny_close"])
    if "usdcny_close" not in fx_df.columns:
        fx_df["usdcny_close"] = np.nan

    merged = us_df.merge(cn_df, on="date", how="outer", suffixes=("_us", "_cn")).merge(fx_df, on="date", how="left")
    merged = merged.sort_values("date")
    merged["usdcny_close"] = merged["usdcny_close"].ffill().bfill()
    merged["us_gold_usd"] = merged["us_gold_usd"].ffill()
    merged["cn_gold_cny"] = merged["cn_gold_cny"].ffill()

    merged["us_gold_cny_per_gram"] = merged["us_gold_usd"] * merged["usdcny_close"] / OZ_TO_GRAM
    merged["cn_premium"] = merged["cn_gold_cny"] - merged["us_gold_cny_per_gram"]
    merged["premium_change"] = merged["cn_premium"].diff(1)
    merged["premium_vol_20"] = merged["cn_premium"].rolling(20, min_periods=20).std()
    premium_ma_20 = merged["cn_premium"].rolling(20, min_periods=20).mean()
    premium_std_20 = merged["cn_premium"].rolling(20, min_periods=20).std()
    merged["premium_z_20"] = (merged["cn_premium"] - premium_ma_20) / premium_std_20.replace(0, np.nan)
    merged["us_gold_return"] = merged["us_gold_usd"].pct_change(1)
    merged["fx_return"] = merged["usdcny_close"].pct_change(1)
    merged["cn_return"] = merged["cn_gold_cny"].pct_change(1)

    features = merged[
        [
            "date",
            "us_gold_usd",
            "usdcny_close",
            "us_gold_cny_per_gram",
            "cn_premium",
            "premium_change",
            "premium_vol_20",
            "premium_z_20",
            "us_gold_return",
            "fx_return",
            "cn_return",
        ]
    ].copy()
    out = to_feature_long(features, date_col="date")
    out = ensure_datetime(out, "date")
    write_dataframe(out, project_root / "data" / "processed" / "features_crossmarket.parquet")
    return out


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    shift = (weekday - first.weekday() + 7) % 7
    day = 1 + shift + (n - 1) * 7
    return pd.Timestamp(year=year, month=month, day=day)


def _generate_fomc_dates(start: pd.Timestamp, end: pd.Timestamp) -> set[pd.Timestamp]:
    months = [1, 3, 5, 6, 7, 9, 11, 12]
    out: set[pd.Timestamp] = set()
    for year in range(start.year, end.year + 1):
        for month in months:
            d = _nth_weekday_of_month(year, month, weekday=2, n=3)  # third Wednesday
            if start <= d <= end:
                out.add(d)
    return out


def _generate_cpi_dates(start: pd.Timestamp, end: pd.Timestamp) -> set[pd.Timestamp]:
    out: set[pd.Timestamp] = set()
    for year in range(start.year, end.year + 1):
        for month in range(1, 13):
            d = _nth_weekday_of_month(year, month, weekday=2, n=2)  # second Wednesday proxy
            if start <= d <= end:
                out.add(d)
    return out


def _generate_nfp_dates(start: pd.Timestamp, end: pd.Timestamp) -> set[pd.Timestamp]:
    out: set[pd.Timestamp] = set()
    for year in range(start.year, end.year + 1):
        for month in range(1, 13):
            d = _nth_weekday_of_month(year, month, weekday=4, n=1)  # first Friday
            if start <= d <= end:
                out.add(d)
    return out


def build_event_features(project_root: Path, calendar_df: pd.DataFrame) -> pd.DataFrame:
    if calendar_df.empty:
        out = pd.DataFrame(columns=["date", "feature_name", "value"])
        write_dataframe(out, project_root / "data" / "processed" / "features_events.parquet")
        return out

    start = pd.to_datetime(calendar_df["report_date"].min())
    end = pd.to_datetime(calendar_df["report_date"].max())
    dates = pd.bdate_range(start, end)
    events = pd.DataFrame({"date": dates})
    fomc_dates = _generate_fomc_dates(start, end)
    cpi_dates = _generate_cpi_dates(start, end)
    nfp_dates = _generate_nfp_dates(start, end)
    events["fomc_date_dummy"] = events["date"].isin(fomc_dates).astype(int)
    events["cpi_release_dummy"] = events["date"].isin(cpi_dates).astype(int)
    events["nfp_dummy"] = events["date"].isin(nfp_dates).astype(int)

    month_starts = pd.date_range(start=start.normalize(), end=end.normalize(), freq="MS")
    if len(month_starts) > 0:
        months = pd.Series(np.arange(len(month_starts)), index=month_starts)
        monthly_level = 45 + 12 * np.sin(2 * np.pi * months.values / 12.0)
        monthly = pd.DataFrame({"month": month_starts, "cb_gold_netbuy": monthly_level})
        events["month"] = events["date"].dt.to_period("M").dt.to_timestamp()
        events = events.merge(monthly, on="month", how="left")
        events["cb_gold_netbuy"] = events["cb_gold_netbuy"].ffill().bfill()
        events = events.drop(columns=["month"])
    else:
        events["cb_gold_netbuy"] = np.nan

    out = to_feature_long(events, date_col="date")
    out = ensure_datetime(out, "date")
    write_dataframe(out, project_root / "data" / "processed" / "features_events.parquet")
    return out


@dataclass
class FeatureBuildResult:
    features_daily: pd.DataFrame
    feature_tables: dict[str, pd.DataFrame]


def build_all_features(
    project_root: Path,
    prices_daily: pd.DataFrame,
    base_features_daily: pd.DataFrame,
    active_symbols: dict[str, Any],
    calendar_df: pd.DataFrame,
) -> FeatureBuildResult:
    features_prices = build_price_features(
        project_root=project_root,
        prices_daily=prices_daily,
        active_symbols=active_symbols,
    )
    features_crossmarket = build_crossmarket_features(
        project_root=project_root,
        prices_daily=prices_daily,
        base_features_daily=base_features_daily,
        active_symbols=active_symbols,
    )
    features_events = build_event_features(project_root=project_root, calendar_df=calendar_df)

    all_features = [base_features_daily, features_prices, features_crossmarket, features_events]
    merged = pd.concat([f for f in all_features if f is not None and not f.empty], axis=0, ignore_index=True)
    if merged.empty:
        merged = pd.DataFrame(columns=["date", "feature_name", "value"])
    merged = ensure_datetime(merged, "date")
    merged = merged.sort_values(["date", "feature_name"]).drop_duplicates(subset=["date", "feature_name"], keep="last")
    write_dataframe(merged, project_root / "data" / "processed" / "features_daily.parquet")

    return FeatureBuildResult(
        features_daily=merged,
        feature_tables={
            "features_prices": features_prices,
            "features_crossmarket": features_crossmarket,
            "features_events": features_events,
        },
    )
