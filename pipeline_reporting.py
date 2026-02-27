from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment

from pipeline_io import ensure_datetime


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Daily Gold Forecast Report - {{ report_date }}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; color: #111; }
    h1,h2,h3 { margin-bottom: 8px; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 18px; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background: #f5f5f5; }
    .row { display: flex; gap: 16px; }
    .card { flex: 1; border: 1px solid #ddd; padding: 12px; border-radius: 6px; }
    .muted { color: #666; font-size: 12px; }
    .alert { color: #9f0000; }
  </style>
</head>
<body>
  <h1>Daily Gold Forecast Report</h1>
  <p class="muted">Report Date: {{ report_date }}</p>

  <h2>Data Status</h2>
  <table>
    <thead><tr><th>Market</th><th>Symbol</th><th>Latest Date</th><th>Source</th><th>Synthetic</th></tr></thead>
    <tbody>
      {% for r in data_status_rows %}
      <tr>
        <td>{{ r.market }}</td>
        <td>{{ r.symbol }}</td>
        <td>{{ r.latest_date }}</td>
        <td>{{ r.source }}</td>
        <td>{{ r.synthetic }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <div class="row">
    <div class="card">
      <h3>US Forecast</h3>
      <p>1D return point: {{ us.next_1d_return_point }}</p>
      <p>Up probability: {{ us.direction_prob_up }}</p>
      <p>Interval P10/P90: {{ us.interval_p10 }} / {{ us.interval_p90 }}</p>
      <p>Vol forecast: {{ us.vol_forecast }}</p>
      <p class="muted">Top drivers: {{ us_top_drivers }}</p>
    </div>
    <div class="card">
      <h3>CN Forecast</h3>
      <p>1D return point: {{ cn.next_1d_return_point }}</p>
      <p>Up probability: {{ cn.direction_prob_up }}</p>
      <p>Interval P10/P90: {{ cn.interval_p10 }} / {{ cn.interval_p90 }}</p>
      <p>Vol forecast: {{ cn.vol_forecast }}</p>
      <p class="muted">Top drivers: {{ cn_top_drivers }}</p>
    </div>
  </div>

  <h2>CN Decomposition (US / FX / Premium)</h2>
  <p>US contribution: {{ cn_decomp.US_gold }}</p>
  <p>FX contribution: {{ cn_decomp.FX_usdcny }}</p>
  <p>Premium contribution: {{ cn_decomp.CN_premium }}</p>

  <h2>Risk Flags</h2>
  <ul>
    {% for a in alerts %}
    <li class="{{ 'alert' if a.is_alert else '' }}">{{ a.text }}</li>
    {% endfor %}
  </ul>

  <h2>Charts</h2>
  <img src="{{ charts.price_trend }}" style="max-width: 100%;" />
  <img src="{{ charts.interval_band }}" style="max-width: 100%;" />
  <img src="{{ charts.cn_decomposition }}" style="max-width: 100%;" />

  <h2>Appendix</h2>
  <p>See backtest report and monitoring report for details.</p>
</body>
</html>
"""


def _fmt(x: float | int | None, nd: int = 6) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _extract_series(prices_daily: pd.DataFrame, market: str, symbol: str) -> pd.DataFrame:
    df = prices_daily.loc[(prices_daily["market"] == market) & (prices_daily["symbol"] == symbol), ["date", "close"]].copy()
    if df.empty:
        return df
    return df.sort_values("date").drop_duplicates(subset=["date"], keep="last")


def _draw_price_trend(
    out_path: Path,
    us_series: pd.DataFrame,
    cn_series: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if not us_series.empty:
        p = us_series.tail(180)
        ax.plot(p["date"], p["close"], label="US Gold")
    if not cn_series.empty:
        p = cn_series.tail(180)
        ax.plot(p["date"], p["close"], label="CN Gold")
    ax.set_title("US and CN Gold Price Trend")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _draw_interval_band(out_path: Path, asof: pd.Timestamp, us_pred: dict[str, Any], cn_pred: dict[str, Any]) -> None:
    def _num(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    fig, ax = plt.subplots(figsize=(10, 4))
    labels = ["US", "CN"]
    point = [_num(us_pred.get("next_1d_return_point")), _num(cn_pred.get("next_1d_return_point"))]
    low = [_num(us_pred.get("interval_p10")), _num(cn_pred.get("interval_p10"))]
    high = [_num(us_pred.get("interval_p90")), _num(cn_pred.get("interval_p90"))]
    x = [0, 1]
    ax.scatter(x, point, color="black", label="Point forecast")
    for i in range(2):
        ax.vlines(x[i], low[i], high[i], color="tab:blue", linewidth=5, alpha=0.6)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_title(f"Next 1D Return Forecast Interval ({pd.to_datetime(asof).date()})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _draw_cn_contrib(out_path: Path, contributions: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = ["US_gold", "FX_usdcny", "CN_premium"]
    vals = [float(contributions.get(k, 0.0)) for k in keys]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    ax.bar(keys, vals, color=colors)
    ax.axhline(0.0, color="gray", linewidth=1)
    ax.set_title("CN Return Contribution Decomposition")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _top_drivers(features_daily: pd.DataFrame, prefixes: list[str], asof: pd.Timestamp, top_n: int = 5) -> str:
    if features_daily.empty:
        return "N/A"
    df = ensure_datetime(features_daily, "date")
    latest = df.loc[df["date"] <= asof].sort_values("date").tail(300)
    if latest.empty:
        return "N/A"
    pivot = latest.pivot_table(index="date", columns="feature_name", values="value", aggfunc="last")
    if pivot.empty:
        return "N/A"
    row = pivot.tail(1).T
    scores = []
    for feature in row.index:
        if prefixes and not any(str(feature).startswith(p) for p in prefixes):
            continue
        value = row.loc[feature].iloc[0]
        if pd.isna(value):
            continue
        scores.append((feature, abs(float(value)), float(value)))
    if not scores:
        return "N/A"
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    return ", ".join([f"{s[0]}={s[2]:.4f}" for s in scores])


def _build_alerts(monitoring_text: str) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for line in monitoring_text.splitlines():
        if not line.strip().startswith("-"):
            continue
        txt = line.strip("- ").strip()
        is_alert = ("HIGH" in txt) or ("Synthetic fallback" in txt) or ("jump_alert=True" in txt)
        alerts.append({"text": txt, "is_alert": is_alert})
    return alerts


def _build_data_status_rows(
    data_status: dict[str, Any],
    active_symbols: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for mk, market in [("us", "US"), ("cn", "CN")]:
        rows.append(
            {
                "market": market,
                "symbol": active_symbols.get(mk, {}).get("symbol", "N/A"),
                "latest_date": data_status.get(f"{mk}_latest", "N/A"),
                "source": data_status.get(mk, {}).get("chosen_source", "N/A"),
                "synthetic": str(bool(data_status.get(mk, {}).get("used_synthetic", False))),
            }
        )
    return rows


def generate_daily_report(
    project_root: Path,
    asof: pd.Timestamp,
    prices_daily: pd.DataFrame,
    features_daily: pd.DataFrame,
    market_results: dict[str, Any],
    decomposition: dict[str, Any],
    data_status: dict[str, Any],
    monitoring_report_path: Path,
    active_symbols: dict[str, Any],
) -> tuple[Path, Path]:
    report_date = pd.to_datetime(asof).date().isoformat()
    daily_dir = project_root / "reports" / "daily"
    assets_dir = daily_dir / "assets"
    daily_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    us_symbol = active_symbols.get("us", {}).get("symbol")
    cn_symbol = active_symbols.get("cn", {}).get("symbol")
    us_series = _extract_series(prices_daily, "US", us_symbol) if us_symbol else pd.DataFrame()
    cn_series = _extract_series(prices_daily, "CN", cn_symbol) if cn_symbol else pd.DataFrame()

    price_chart = assets_dir / f"{report_date}_price_trend.png"
    interval_chart = assets_dir / f"{report_date}_interval_band.png"
    decomp_chart = assets_dir / f"{report_date}_cn_decomp.png"

    _draw_price_trend(price_chart, us_series, cn_series)
    us_pred = market_results.get("us", {}).get("prediction", {})
    cn_pred = market_results.get("cn", {}).get("prediction", {})
    _draw_interval_band(interval_chart, asof, us_pred, cn_pred)
    _draw_cn_contrib(decomp_chart, decomposition.get("contributions", {}))

    monitoring_text = monitoring_report_path.read_text(encoding="utf-8") if monitoring_report_path.exists() else ""
    alerts = _build_alerts(monitoring_text)
    data_rows = _build_data_status_rows(data_status=data_status, active_symbols=active_symbols)
    us_top_drivers = _top_drivers(features_daily, prefixes=["us_", "dxy", "us10y", "vix"], asof=asof)
    cn_top_drivers = _top_drivers(features_daily, prefixes=["cn_", "premium", "fx_", "usdcny"], asof=asof)

    env = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(HTML_TEMPLATE)
    html = template.render(
        report_date=report_date,
        data_status_rows=data_rows,
        us={k: _fmt(v) if isinstance(v, (int, float)) else v for k, v in us_pred.items()},
        cn={k: _fmt(v) if isinstance(v, (int, float)) else v for k, v in cn_pred.items()},
        cn_decomp={k: _fmt(v) if isinstance(v, (int, float)) else v for k, v in decomposition.get("contributions", {}).items()},
        us_top_drivers=us_top_drivers,
        cn_top_drivers=cn_top_drivers,
        alerts=alerts,
        charts={
            "price_trend": str(price_chart.name if price_chart.parent == daily_dir else f"assets/{price_chart.name}"),
            "interval_band": str(interval_chart.name if interval_chart.parent == daily_dir else f"assets/{interval_chart.name}"),
            "cn_decomposition": str(decomp_chart.name if decomp_chart.parent == daily_dir else f"assets/{decomp_chart.name}"),
        },
    )

    html_path = daily_dir / f"{report_date}_gold_report.html"
    md_path = daily_dir / f"{report_date}_gold_report.md"
    html_path.write_text(html, encoding="utf-8")

    md_lines = [
        f"# Daily Gold Forecast Report ({report_date})",
        "",
        "## US Forecast",
        f"- point return: {_fmt(us_pred.get('next_1d_return_point'))}",
        f"- up probability: {_fmt(us_pred.get('direction_prob_up'))}",
        f"- interval p10/p90: {_fmt(us_pred.get('interval_p10'))} / {_fmt(us_pred.get('interval_p90'))}",
        "",
        "## CN Forecast",
        f"- point return: {_fmt(cn_pred.get('next_1d_return_point'))}",
        f"- up probability: {_fmt(cn_pred.get('direction_prob_up'))}",
        f"- interval p10/p90: {_fmt(cn_pred.get('interval_p10'))} / {_fmt(cn_pred.get('interval_p90'))}",
        "",
        "## CN Decomposition",
        f"- US contribution: {_fmt(decomposition.get('contributions', {}).get('US_gold'))}",
        f"- FX contribution: {_fmt(decomposition.get('contributions', {}).get('FX_usdcny'))}",
        f"- Premium contribution: {_fmt(decomposition.get('contributions', {}).get('CN_premium'))}",
        "",
        "## Risk Flags",
    ]
    for a in alerts:
        md_lines.append(f"- {a['text']}")
    md_path.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")
    return html_path, md_path
