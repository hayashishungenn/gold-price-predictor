from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline_io import ensure_datetime, write_dataframe


@dataclass
class CalendarBuildResult:
    calendar: pd.DataFrame
    warnings: list[str]


def _market_dates(prices_daily: pd.DataFrame, market: str, symbol: str | None = None) -> pd.DataFrame:
    df = prices_daily.loc[prices_daily["market"] == market, ["date", "symbol"]].copy()
    if symbol is not None:
        df = df.loc[df["symbol"] == symbol]
    if df.empty:
        return pd.DataFrame(columns=["date"])
    dates = pd.DataFrame({"date": pd.to_datetime(df["date"]).drop_duplicates().sort_values()})
    return dates


def build_calendar(
    project_root: Path,
    prices_daily: pd.DataFrame,
    us_symbol: str | None = None,
    cn_symbol: str | None = None,
) -> CalendarBuildResult:
    prices = ensure_datetime(prices_daily, "date")
    us_dates = _market_dates(prices, "US", us_symbol)
    cn_dates = _market_dates(prices, "CN", cn_symbol)
    warnings: list[str] = []

    if us_dates.empty:
        warnings.append("US series is empty; calendar US alignment unavailable.")
    if cn_dates.empty:
        warnings.append("CN series is empty; calendar CN alignment unavailable.")

    if us_dates.empty and cn_dates.empty:
        out = pd.DataFrame(columns=["report_date", "us_trade_date", "cn_trade_date", "us_lag_days", "cn_lag_days"])
        write_dataframe(out, project_root / "data" / "processed" / "calendar.parquet")
        return CalendarBuildResult(calendar=out, warnings=warnings)

    min_date = min(
        us_dates["date"].min() if not us_dates.empty else cn_dates["date"].min(),
        cn_dates["date"].min() if not cn_dates.empty else us_dates["date"].min(),
    )
    max_date = max(
        us_dates["date"].max() if not us_dates.empty else cn_dates["date"].max(),
        cn_dates["date"].max() if not cn_dates.empty else us_dates["date"].max(),
    )
    report_dates = pd.DataFrame({"report_date": pd.bdate_range(min_date, max_date)})

    if not us_dates.empty:
        us_aligned = pd.merge_asof(
            report_dates.sort_values("report_date"),
            us_dates.rename(columns={"date": "us_trade_date"}).sort_values("us_trade_date"),
            left_on="report_date",
            right_on="us_trade_date",
            direction="backward",
        )
    else:
        us_aligned = report_dates.copy()
        us_aligned["us_trade_date"] = pd.NaT

    if not cn_dates.empty:
        cn_aligned = pd.merge_asof(
            us_aligned.sort_values("report_date"),
            cn_dates.rename(columns={"date": "cn_trade_date"}).sort_values("cn_trade_date"),
            left_on="report_date",
            right_on="cn_trade_date",
            direction="backward",
        )
    else:
        cn_aligned = us_aligned.copy()
        cn_aligned["cn_trade_date"] = pd.NaT

    out = cn_aligned.copy()
    out["us_lag_days"] = (out["report_date"] - out["us_trade_date"]).dt.days
    out["cn_lag_days"] = (out["report_date"] - out["cn_trade_date"]).dt.days
    out["alignment_note"] = ""
    out.loc[out["us_lag_days"] > 2, "alignment_note"] += "US_STALE;"
    out.loc[out["cn_lag_days"] > 2, "alignment_note"] += "CN_STALE;"
    out["alignment_note"] = out["alignment_note"].str.strip(";")

    mismatch = out.loc[(out["us_trade_date"].isna()) | (out["cn_trade_date"].isna())]
    if len(mismatch) > 0:
        warnings.append(f"Calendar has {len(mismatch)} rows with missing US/CN trade date mapping.")

    write_dataframe(out, project_root / "data" / "processed" / "calendar.parquet")
    return CalendarBuildResult(calendar=out, warnings=warnings)
