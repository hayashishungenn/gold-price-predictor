from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline_io import ensure_datetime, from_feature_long


def _calc_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float | None:
    ref = reference.dropna().astype(float)
    cur = current.dropna().astype(float)
    if len(ref) < bins * 2 or len(cur) < bins * 2:
        return None
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 4:
        return None
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-6, 1)
    cur_pct = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-6, 1)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def generate_monitoring_report(
    project_root: Path,
    asof: pd.Timestamp,
    prices_daily: pd.DataFrame,
    features_daily: pd.DataFrame,
    active_symbols: dict[str, Any],
    data_status: dict[str, Any],
) -> Path:
    prices = ensure_datetime(prices_daily, "date")
    features = ensure_datetime(features_daily, "date")
    feature_wide = from_feature_long(features) if not features.empty else pd.DataFrame(columns=["date"])

    lines = ["# Monitoring Daily", "", f"- asof: {pd.to_datetime(asof).date()}", ""]
    lines.append("## Data Quality")
    for mk, market in [("us", "US"), ("cn", "CN")]:
        symbol = active_symbols.get(mk, {}).get("symbol")
        if not symbol:
            lines.append(f"- {market}: missing active symbol")
            continue
        series = prices.loc[(prices["market"] == market) & (prices["symbol"] == symbol)].sort_values("date")
        if series.empty:
            lines.append(f"- {market}/{symbol}: no records")
            continue
        missing_rate = float(series["close"].isna().mean())
        latest = pd.to_datetime(series["date"].max()).date()
        returns = series["close"].pct_change()
        jump_flag = bool((returns.abs() > 0.08).tail(5).any())
        lines.append(
            f"- {market}/{symbol}: latest={latest}, missing_rate={missing_rate:.4f}, recent_jump_alert={jump_flag}, source={data_status.get(mk, {}).get('chosen_source')}"
        )

    lines.append("")
    lines.append("## Drift")
    if feature_wide.empty:
        lines.append("- No feature table available.")
    else:
        feature_wide = feature_wide.sort_values("date")
        recent = feature_wide.tail(60)
        reference = feature_wide.iloc[max(0, len(feature_wide) - 320) : max(0, len(feature_wide) - 60)]
        candidate_cols = [c for c in feature_wide.columns if c != "date"]
        drift_rows = []
        for col in candidate_cols[:30]:
            psi = _calc_psi(reference[col], recent[col])
            if psi is None:
                continue
            drift_rows.append((col, psi))
        drift_rows = sorted(drift_rows, key=lambda x: x[1], reverse=True)
        if not drift_rows:
            lines.append("- Insufficient data for PSI computation.")
        else:
            for col, psi in drift_rows[:10]:
                level = "HIGH" if psi >= 0.25 else ("MEDIUM" if psi >= 0.1 else "LOW")
                lines.append(f"- {col}: PSI={psi:.4f} ({level})")

    lines.append("")
    lines.append("## Alerts")
    us_latest = data_status.get("us_latest")
    cn_latest = data_status.get("cn_latest")
    lines.append(f"- US latest timestamp: {us_latest}")
    lines.append(f"- CN latest timestamp: {cn_latest}")
    if data_status.get("us", {}).get("used_synthetic") or data_status.get("cn", {}).get("used_synthetic"):
        lines.append("- Synthetic fallback used for at least one market (check data source availability).")

    content = "\n".join(lines).strip() + "\n"
    out_path = project_root / "reports" / "monitoring" / "monitoring_daily.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path
