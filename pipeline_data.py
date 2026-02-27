from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from pipeline_io import ensure_datetime, to_feature_long, write_dataframe


LOGGER = logging.getLogger(__name__)
OZ_TO_GRAM = 31.1034768


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            parsed = _safe_float(item)
            if parsed is not None:
                return parsed
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)
    cleaned = str(value).replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def normalize_prices(
    raw_df: pd.DataFrame,
    market: str,
    symbol: str,
    currency: str,
    unit: str,
    source: str,
) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    df = _coerce_columns(raw_df)
    date_col = _first_existing(df, ["date", "d", "day", "time", "datetime", "trade_date", "日期"])
    close_col = _first_existing(df, ["close", "settle", "last", "price", "v", "收盘", "收盘价"])
    open_col = _first_existing(df, ["open", "开盘", "开盘价"])
    high_col = _first_existing(df, ["high", "最高", "最高价"])
    low_col = _first_existing(df, ["low", "最低", "最低价"])
    volume_col = _first_existing(df, ["volume", "vol", "成交量", "持仓量"])

    if date_col is None or close_col is None:
        LOGGER.warning("Unable to normalize %s (%s): missing date/close columns", symbol, source)
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None),
            "open": pd.to_numeric(df[open_col], errors="coerce") if open_col else np.nan,
            "high": pd.to_numeric(df[high_col], errors="coerce") if high_col else np.nan,
            "low": pd.to_numeric(df[low_col], errors="coerce") if low_col else np.nan,
            "close": pd.to_numeric(df[close_col], errors="coerce"),
            "volume": pd.to_numeric(df[volume_col], errors="coerce") if volume_col else np.nan,
            "market": market,
            "symbol": symbol,
            "currency": currency,
            "unit": unit,
            "source": source,
        }
    )
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out = out.drop_duplicates(subset=["date", "market", "symbol"], keep="last")
    return out


@dataclass
class PriceSeriesConfig:
    name: str
    market: str
    symbol: str
    source: str
    currency: str
    unit: str


class BaseAdapter:
    source_name: str

    def fetch(self, start: date, end: date, source_cfg: dict[str, Any]) -> pd.DataFrame:
        raise NotImplementedError


class LBMAGoldAdapter(BaseAdapter):
    source_name = "lbma"

    def fetch(self, start: date, end: date, source_cfg: dict[str, Any]) -> pd.DataFrame:
        endpoint = source_cfg.get("endpoint", "https://prices.lbma.org.uk/json/gold_pm.json")
        response = requests.get(endpoint, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            d_raw = item.get("d") or item.get("date")
            v_raw = (
                item.get("v")
                or item.get("value")
                or item.get("USD")
                or item.get("usd")
                or item.get("pm")
                or item.get("PM")
            )
            if isinstance(v_raw, list):
                # LBMA payload format: [USD, GBP, EUR]
                v_raw = v_raw[0] if v_raw else None
            d_value = pd.to_datetime(d_raw, errors="coerce")
            v_value = _safe_float(v_raw)
            if pd.isna(d_value) or v_value is None:
                continue
            rows.append({"date": d_value, "close": v_value})
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
        return df.loc[mask].copy()


class YahooAdapter(BaseAdapter):
    source_name = "yahoo"

    def __init__(self, ticker_key: str = "ticker") -> None:
        self.ticker_key = ticker_key

    def fetch(self, start: date, end: date, source_cfg: dict[str, Any]) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception:
            LOGGER.warning("yfinance not available, skip yahoo fetch")
            return pd.DataFrame()

        ticker = source_cfg.get(self.ticker_key)
        if not ticker:
            return pd.DataFrame()
        try:
            hist = yf.download(
                ticker,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                progress=False,
                interval="1d",
            )
        except Exception as exc:
            LOGGER.warning("yahoo fetch failed for %s: %s", ticker, exc)
            return pd.DataFrame()

        if hist.empty:
            return pd.DataFrame()
        hist = hist.reset_index()
        hist.columns = [str(c).strip().lower() for c in hist.columns]
        if "adj close" in hist.columns and "close" not in hist.columns:
            hist["close"] = hist["adj close"]
        return hist


class SGEAdapter(BaseAdapter):
    source_name = "sge"

    def fetch(self, start: date, end: date, source_cfg: dict[str, Any]) -> pd.DataFrame:
        try:
            import akshare as ak
        except Exception:
            LOGGER.warning("akshare not available, skip SGE source")
            return pd.DataFrame()

        symbol = source_cfg.get("symbol", "Au99.99")
        candidates = [
            ("spot_hist_sge", {"symbol": symbol}),
            ("spot_commodity_hist_sge", {"symbol": symbol}),
        ]
        for fn_name, kwargs in candidates:
            fn = getattr(ak, fn_name, None)
            if fn is None:
                continue
            try:
                frame = fn(**kwargs)
                if frame is None or len(frame) == 0:
                    continue
                df = _coerce_columns(pd.DataFrame(frame))
                date_col = _first_existing(df, ["date", "日期"])
                if date_col is None:
                    continue
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=[date_col])
                mask = (df[date_col].dt.date >= start) & (df[date_col].dt.date <= end)
                return df.loc[mask].copy()
            except Exception as exc:
                LOGGER.warning("SGE adapter call failed (%s): %s", fn_name, exc)
        return pd.DataFrame()


class SHFEAdapter(BaseAdapter):
    source_name = "shfe"

    def fetch(self, start: date, end: date, source_cfg: dict[str, Any]) -> pd.DataFrame:
        try:
            import akshare as ak
        except Exception:
            LOGGER.warning("akshare not available, skip SHFE source")
            return pd.DataFrame()

        preferred_symbols = [source_cfg.get("symbol", "AU0"), "AU9999", "AU"]
        fn = getattr(ak, "futures_zh_daily_sina", None)
        if fn is None:
            return pd.DataFrame()
        for symbol in preferred_symbols:
            try:
                frame = fn(symbol=symbol)
                if frame is None or len(frame) == 0:
                    continue
                df = _coerce_columns(pd.DataFrame(frame))
                date_col = _first_existing(df, ["date", "日期"])
                if date_col is None:
                    continue
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=[date_col])
                mask = (df[date_col].dt.date >= start) & (df[date_col].dt.date <= end)
                return df.loc[mask].copy()
            except Exception as exc:
                LOGGER.warning("SHFE adapter failed for %s: %s", symbol, exc)
        return pd.DataFrame()


class FXAdapter(YahooAdapter):
    source_name = "fx"

    def fetch(self, start: date, end: date, source_cfg: dict[str, Any]) -> pd.DataFrame:
        frame = super().fetch(start=start, end=end, source_cfg=source_cfg)
        if frame.empty:
            frame = self._fetch_frankfurter(start=start, end=end)
            if frame.empty:
                return frame
            return frame
        return frame.rename(columns={"close": "usdcny_close"})

    def _fetch_frankfurter(self, start: date, end: date) -> pd.DataFrame:
        endpoint = f"https://api.frankfurter.app/{start.isoformat()}..{end.isoformat()}"
        params = {"from": "USD", "to": "CNY"}
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            LOGGER.warning("Frankfurter FX fallback failed: %s", exc)
            return pd.DataFrame()
        rates = payload.get("rates", {})
        if not isinstance(rates, dict):
            return pd.DataFrame()
        rows = []
        for d, fx_map in rates.items():
            if not isinstance(fx_map, dict):
                continue
            value = fx_map.get("CNY")
            parsed = _safe_float(value)
            if parsed is None:
                continue
            rows.append({"date": pd.to_datetime(d, errors="coerce"), "usdcny_close": parsed})
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out = out.dropna(subset=["date", "usdcny_close"]).sort_values("date")
        out = out.drop_duplicates(subset=["date"], keep="last")
        return out


class MacroAdapter(BaseAdapter):
    source_name = "macro"

    def fetch(self, start: date, end: date, source_cfg: dict[str, Any]) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception:
            LOGGER.warning("yfinance not available, skip macro source")
            return pd.DataFrame()

        tickers: dict[str, str] = source_cfg.get("tickers", {})
        if not tickers:
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        for name, ticker in tickers.items():
            try:
                hist = yf.download(
                    ticker,
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),
                    auto_adjust=False,
                    progress=False,
                    interval="1d",
                )
            except Exception as exc:
                LOGGER.warning("macro ticker fetch failed %s/%s: %s", name, ticker, exc)
                continue
            if hist.empty:
                continue
            hist = hist.reset_index()
            hist.columns = [str(c).strip().lower() for c in hist.columns]
            if "adj close" in hist.columns and "close" not in hist.columns:
                hist["close"] = hist["adj close"]
            if "date" not in hist.columns or "close" not in hist.columns:
                continue
            f = hist[["date", "close"]].copy()
            f = f.rename(columns={"close": name})
            frames.append(f)
        if not frames:
            return pd.DataFrame()
        out = frames[0]
        for frame in frames[1:]:
            out = out.merge(frame, on="date", how="outer")
        return out.sort_values("date")


def generate_synthetic_prices(
    market: str,
    symbol: str,
    currency: str,
    unit: str,
    start: date,
    end: date,
    seed: int,
) -> pd.DataFrame:
    index = pd.bdate_range(start=start, end=end)
    if len(index) == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    base = 2000.0 if market == "US" else 480.0
    daily_drift = 0.02 / 252.0
    daily_vol = 0.16 / np.sqrt(252.0)
    rets = rng.normal(daily_drift, daily_vol, size=len(index))
    close = base * np.exp(np.cumsum(rets))
    out = pd.DataFrame(
        {
            "date": index,
            "open": np.concatenate(([close[0]], close[:-1])),
            "high": close * (1 + np.abs(rng.normal(0.001, 0.002, size=len(index)))),
            "low": close * (1 - np.abs(rng.normal(0.001, 0.002, size=len(index)))),
            "close": close,
            "volume": rng.integers(1_000, 5_000, size=len(index)).astype(float),
            "market": market,
            "symbol": symbol,
            "currency": currency,
            "unit": unit,
            "source": "synthetic",
        }
    )
    return out


def generate_synthetic_fx(start: date, end: date, seed: int = 7) -> pd.DataFrame:
    index = pd.bdate_range(start=start, end=end)
    if len(index) == 0:
        return pd.DataFrame(columns=["date", "usdcny_close"])
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.03 / np.sqrt(252.0), size=len(index))
    level = 7.2 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"date": index, "usdcny_close": level})


def generate_synthetic_macro(start: date, end: date, seed: int = 13) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, end=end)
    if len(idx) == 0:
        return pd.DataFrame(columns=["date", "dxy", "us10y", "vix", "spx", "oil"])
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "date": idx,
            "dxy": 103 + rng.normal(0, 0.4, size=len(idx)).cumsum() * 0.03,
            "us10y": 4.0 + rng.normal(0, 0.02, size=len(idx)),
            "vix": np.clip(16 + rng.normal(0, 2, size=len(idx)), 10, 45),
            "spx": 4800 + rng.normal(0, 10, size=len(idx)).cumsum(),
            "oil": 75 + rng.normal(0, 1.2, size=len(idx)).cumsum() * 0.2,
        }
    )


def build_adapter(source_name: str) -> BaseAdapter:
    source_name = source_name.lower()
    if source_name == "lbma":
        return LBMAGoldAdapter()
    if source_name == "xauusd":
        return YahooAdapter()
    if source_name == "sge":
        return SGEAdapter()
    if source_name == "shfe":
        return SHFEAdapter()
    if source_name == "fx":
        return FXAdapter()
    if source_name == "macro":
        return MacroAdapter()
    return YahooAdapter()


def _to_series_config(entry: dict[str, Any]) -> PriceSeriesConfig:
    return PriceSeriesConfig(
        name=entry["name"],
        market=str(entry["market"]).upper(),
        symbol=entry["symbol"],
        source=entry["source"],
        currency=entry["currency"],
        unit=entry["unit"],
    )


def _save_raw_snapshot(df: pd.DataFrame, project_root: Path, source: str, asof: date) -> None:
    out_path = project_root / "data" / "raw" / source / f"{asof.strftime('%Y%m%d')}.parquet"
    write_dataframe(df, out_path)


def _fetch_price_candidate(
    cfg: PriceSeriesConfig,
    source_cfg: dict[str, Any],
    start: date,
    end: date,
    project_root: Path,
    asof: date,
) -> pd.DataFrame:
    adapter = build_adapter(cfg.source)
    raw = adapter.fetch(start=start, end=end, source_cfg=source_cfg)
    if raw.empty:
        return raw
    _save_raw_snapshot(raw, project_root=project_root, source=cfg.source, asof=asof)
    return normalize_prices(
        raw_df=raw,
        market=cfg.market,
        symbol=cfg.symbol,
        currency=cfg.currency,
        unit=cfg.unit,
        source=cfg.source,
    )


def _choose_and_fetch_market_series(
    market_key: str,
    symbols_cfg: dict[str, Any],
    sources_cfg: dict[str, Any],
    start: date,
    end: date,
    project_root: Path,
    asof: date,
    allow_synthetic: bool,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    primary_entry = symbols_cfg["primary"][market_key]
    fallback_entries = symbols_cfg.get("fallback", {}).get(market_key, [])
    candidates = [primary_entry] + fallback_entries

    all_frames: list[pd.DataFrame] = []
    chosen: dict[str, Any] | None = None
    status: dict[str, Any] = {"market": market_key.upper(), "chosen_source": None, "used_synthetic": False}

    for entry in candidates:
        cfg = _to_series_config(entry)
        source_cfg = sources_cfg.get("adapters", {}).get(cfg.source, {})
        if source_cfg and source_cfg.get("enabled", True) is False:
            continue
        frame = _fetch_price_candidate(
            cfg=cfg,
            source_cfg=source_cfg,
            start=start,
            end=end,
            project_root=project_root,
            asof=asof,
        )
        if frame.empty:
            LOGGER.warning("No data from %s (%s)", cfg.source, cfg.symbol)
            continue
        all_frames.append(frame)
        if chosen is None:
            chosen = {
                "name": cfg.name,
                "market": cfg.market,
                "symbol": cfg.symbol,
                "source": cfg.source,
                "currency": cfg.currency,
                "unit": cfg.unit,
            }
            status["chosen_source"] = cfg.source

    if chosen is None and allow_synthetic:
        cfg = _to_series_config(primary_entry)
        synth = generate_synthetic_prices(
            market=cfg.market,
            symbol=cfg.symbol,
            currency=cfg.currency,
            unit=cfg.unit,
            start=start,
            end=end,
            seed=101 if market_key == "us" else 202,
        )
        all_frames.append(synth)
        chosen = {
            "name": cfg.name,
            "market": cfg.market,
            "symbol": cfg.symbol,
            "source": "synthetic",
            "currency": cfg.currency,
            "unit": cfg.unit,
        }
        status["chosen_source"] = "synthetic"
        status["used_synthetic"] = True
        LOGGER.warning("Using synthetic fallback for %s market", cfg.market)

    if not all_frames:
        return pd.DataFrame(), {}, status

    prices = pd.concat(all_frames, axis=0, ignore_index=True)
    prices = prices.dropna(subset=["date", "close"])
    prices = prices.sort_values(["date", "market", "symbol", "source"])
    prices = prices.drop_duplicates(subset=["date", "market", "symbol", "source"], keep="last")
    return prices, chosen or {}, status


def _fetch_fx_features(
    start: date,
    end: date,
    sources_cfg: dict[str, Any],
    project_root: Path,
    asof: date,
    allow_synthetic: bool,
) -> pd.DataFrame:
    source_cfg = sources_cfg.get("adapters", {}).get("fx", {})
    adapter = FXAdapter()
    raw = adapter.fetch(start=start, end=end, source_cfg=source_cfg)
    if not raw.empty:
        _save_raw_snapshot(raw, project_root=project_root, source="fx", asof=asof)
        raw = _coerce_columns(raw)
        date_col = _first_existing(raw, ["date", "datetime"])
        if date_col and "usdcny_close" in raw.columns:
            out = raw[[date_col, "usdcny_close"]].rename(columns={date_col: "date"}).copy()
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out = out.dropna(subset=["date", "usdcny_close"]).sort_values("date")
            out = out.drop_duplicates(subset=["date"], keep="last")
            return out
    if allow_synthetic:
        return generate_synthetic_fx(start=start, end=end)
    return pd.DataFrame(columns=["date", "usdcny_close"])


def _fetch_macro_features(
    start: date,
    end: date,
    sources_cfg: dict[str, Any],
    project_root: Path,
    asof: date,
    allow_synthetic: bool,
) -> pd.DataFrame:
    source_cfg = sources_cfg.get("adapters", {}).get("macro", {})
    adapter = MacroAdapter()
    raw = adapter.fetch(start=start, end=end, source_cfg=source_cfg)
    if not raw.empty:
        _save_raw_snapshot(raw, project_root=project_root, source="macro", asof=asof)
        raw = _coerce_columns(raw)
        if "date" not in raw.columns:
            return pd.DataFrame()
        out = raw.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date")
        out = out.drop_duplicates(subset=["date"], keep="last")
        return out
    if allow_synthetic:
        return generate_synthetic_macro(start=start, end=end)
    return pd.DataFrame(columns=["date", "dxy", "us10y", "vix", "spx", "oil"])


def ingest_all_data(
    project_root: Path,
    symbols_cfg: dict[str, Any],
    sources_cfg: dict[str, Any],
    asof: date,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    years_history = int(sources_cfg.get("ingestion", {}).get("years_history", 8))
    allow_synthetic = bool(sources_cfg.get("ingestion", {}).get("allow_synthetic_fallback", True))
    start = asof - timedelta(days=int(365.25 * years_history))
    end = asof

    us_prices, chosen_us, us_status = _choose_and_fetch_market_series(
        market_key="us",
        symbols_cfg=symbols_cfg,
        sources_cfg=sources_cfg,
        start=start,
        end=end,
        project_root=project_root,
        asof=asof,
        allow_synthetic=allow_synthetic,
    )
    cn_prices, chosen_cn, cn_status = _choose_and_fetch_market_series(
        market_key="cn",
        symbols_cfg=symbols_cfg,
        sources_cfg=sources_cfg,
        start=start,
        end=end,
        project_root=project_root,
        asof=asof,
        allow_synthetic=allow_synthetic,
    )
    prices_daily = pd.concat([us_prices, cn_prices], axis=0, ignore_index=True)
    prices_daily = ensure_datetime(prices_daily, "date")
    prices_daily = prices_daily.sort_values(["date", "market", "symbol", "source"]).drop_duplicates(
        subset=["date", "market", "symbol", "source"],
        keep="last",
    )

    fx_df = _fetch_fx_features(
        start=start,
        end=end,
        sources_cfg=sources_cfg,
        project_root=project_root,
        asof=asof,
        allow_synthetic=allow_synthetic,
    )
    macro_df = _fetch_macro_features(
        start=start,
        end=end,
        sources_cfg=sources_cfg,
        project_root=project_root,
        asof=asof,
        allow_synthetic=allow_synthetic,
    )

    features_wide = None
    for frame in [fx_df, macro_df]:
        if frame is None or frame.empty:
            continue
        frame = ensure_datetime(frame, "date")
        features_wide = frame if features_wide is None else features_wide.merge(frame, on="date", how="outer")
    if features_wide is None:
        features_wide = pd.DataFrame(columns=["date"])

    features_wide = features_wide.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    features_daily = to_feature_long(features_wide, date_col="date")
    features_daily = ensure_datetime(features_daily, "date")

    write_dataframe(prices_daily, project_root / "data" / "processed" / "prices_daily.parquet")
    write_dataframe(features_daily, project_root / "data" / "processed" / "features_daily.parquet")

    status = {
        "us": us_status,
        "cn": cn_status,
        "us_latest": str(prices_daily.loc[prices_daily["market"] == "US", "date"].max()) if not prices_daily.empty else None,
        "cn_latest": str(prices_daily.loc[prices_daily["market"] == "CN", "date"].max()) if not prices_daily.empty else None,
    }
    active_symbols = {"us": chosen_us, "cn": chosen_cn}
    return prices_daily, features_daily, active_symbols, status
