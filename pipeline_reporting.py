from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment

from pipeline_io import ensure_datetime, from_feature_long


plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


HTML_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{{ report_title }}</title>
  <style>
    :root {
      --bg: #f6f1e7;
      --paper: #fffdf8;
      --ink: #1f1a14;
      --muted: #6f665d;
      --line: #e5d7bf;
      --gold: #b8871b;
      --up: #1f7a45;
      --down: #a33a2b;
      --warn: #a66a00;
      --soft-up: #e6f4ea;
      --soft-down: #fae9e7;
      --soft-warn: #fff4df;
      --card-shadow: 0 10px 24px rgba(80, 58, 22, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at top right, #f2dca7 0, rgba(242, 220, 167, 0) 28%), var(--bg);
      color: var(--ink);
      font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
      line-height: 1.65;
    }
    .container { max-width: 1120px; margin: 0 auto; padding: 28px 20px 40px; }
    .hero, .grid-2, .chart-grid { display: grid; gap: 18px; }
    .hero { grid-template-columns: 1.2fr 0.8fr; margin-bottom: 18px; }
    .grid-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .chart-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .hero-main, .hero-side, .panel {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 22px;
      box-shadow: var(--card-shadow);
    }
    .eyebrow { margin: 0 0 8px; color: var(--gold); font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; }
    h1, h2, h3 { margin: 0 0 10px; line-height: 1.3; }
    h1 { font-size: 30px; }
    h2 { font-size: 22px; margin-top: 28px; }
    h3 { font-size: 17px; }
    p { margin: 0 0 10px; }
    .lead { font-size: 18px; font-weight: 600; }
    .muted { color: var(--muted); font-size: 13px; }
    .badge {
      display: inline-block;
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 12px;
    }
    .positive { background: var(--soft-up); color: var(--up); }
    .negative { background: var(--soft-down); color: var(--down); }
    .warning { background: var(--soft-warn); color: var(--warn); }
    .neutral { background: #f0ece6; color: #5d5348; }
    .kv-list { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 18px; margin: 12px 0 0; }
    .kv-item { padding: 10px 12px; background: #fcfaf5; border: 1px solid #efe3d0; border-radius: 12px; }
    .kv-item .k { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .kv-item .v { display: block; font-size: 16px; font-weight: 700; }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: var(--card-shadow);
    }
    th, td { padding: 12px 14px; border-bottom: 1px solid #efe6d8; vertical-align: top; text-align: left; }
    th { background: #f8f2e7; font-size: 13px; }
    tr:last-child td { border-bottom: none; }
    ul.clean { list-style: none; padding: 0; margin: 10px 0 0; }
    ul.clean li { padding: 10px 12px; border: 1px solid #efe3d0; border-radius: 12px; background: #fcfaf5; margin-bottom: 10px; }
    .signal-item { border-left: 4px solid #d8c39d; }
    .signal-item.positive { border-left-color: #6eb27d; }
    .signal-item.negative { border-left-color: #d2776b; }
    .signal-item.warning { border-left-color: #d8a23a; }
    .chart-grid img {
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #fff;
      box-shadow: var(--card-shadow);
    }
    .chart-caption { margin-top: 8px; color: var(--muted); font-size: 12px; }
    @media (max-width: 900px) {
      .hero, .grid-2, .chart-grid, .kv-list { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="container">
    <section class="hero">
      <div class="hero-main">
        <p class="eyebrow">Gold Daily Analysis</p>
        <h1>{{ report_title }}</h1>
        <p class="muted">报告日期：{{ report_date }} | 数据截至：{{ data_cutoff_text }}</p>
        <p class="lead">{{ core_view.summary }}</p>
        <p>{{ core_view.execution_note }}</p>
        <p class="muted">结论置信度：{{ core_view.confidence_label }} | {{ core_view.freshness_note }}</p>
      </div>
      <div class="hero-side">
        <div class="badge {{ core_view.tone_class }}">{{ core_view.action_tag }}</div>
        <p><strong>一句话执行建议：</strong>{{ core_view.action_line }}</p>
        <p><strong>主导因素：</strong>{{ core_view.dominant_driver }}</p>
        <p><strong>当前约束：</strong>{{ core_view.constraint_line }}</p>
        <p class="muted">格式参考了日度投资决策简报的组织方式：先给一句话判断，再给价格区间、检查清单与风险说明。</p>
      </div>
    </section>

    <section>
      <h2>今日操作计划</h2>
      <table>
        <thead><tr><th>项目</th><th>数值</th><th>说明</th></tr></thead>
        <tbody>
          {% for row in execution_rows %}
          <tr><td>{{ row.label }}</td><td>{{ row.value }}</td><td>{{ row.note }}</td></tr>
          {% endfor %}
        </tbody>
      </table>
    </section>

    <section>
      <h2>操作检查清单</h2>
      <table>
        <thead><tr><th>检查项</th><th>状态</th><th>解读</th></tr></thead>
        <tbody>
          {% for row in checklist_rows %}
          <tr><td>{{ row.label }}</td><td><span class="badge {{ row.status_class }}">{{ row.status }}</span></td><td>{{ row.detail }}</td></tr>
          {% endfor %}
        </tbody>
      </table>
    </section>

    <section>
      <h2>市场状态概览</h2>
      <div class="grid-2">
        {% for card in market_cards %}
        <div class="panel">
          <h3>{{ card.title }}</h3>
          <p class="muted">{{ card.symbol }}</p>
          <div class="kv-list">
            <div class="kv-item"><span class="k">最新价格</span><span class="v">{{ card.latest_close }}</span></div>
            <div class="kv-item"><span class="k">模型点预测</span><span class="v">{{ card.point_return }}</span></div>
            <div class="kv-item"><span class="k">上涨概率</span><span class="v">{{ card.up_probability }}</span></div>
            <div class="kv-item"><span class="k">P10 / P90</span><span class="v">{{ card.interval }}</span></div>
            <div class="kv-item"><span class="k">近1日 / 5日</span><span class="v">{{ card.ret_1d }} / {{ card.ret_5d }}</span></div>
            <div class="kv-item"><span class="k">近20日 / 年内位置</span><span class="v">{{ card.ret_20d }} / {{ card.range_position }}</span></div>
          </div>
          <p class="muted" style="margin-top: 12px;">{{ card.commentary }}</p>
        </div>
        {% endfor %}
      </div>
    </section>

    <section>
      <h2>驱动拆解</h2>
      <table>
        <thead><tr><th>驱动项</th><th>贡献</th><th>解读</th></tr></thead>
        <tbody>
          {% for row in driver_rows %}
          <tr><td>{{ row.label }}</td><td><span class="badge {{ row.tone_class }}">{{ row.value }}</span></td><td>{{ row.note }}</td></tr>
          {% endfor %}
        </tbody>
      </table>
    </section>

    <section>
      <h2>重点信号</h2>
      <div class="grid-2">
        <div class="panel">
          <h3>美盘侧</h3>
          <ul class="clean">
            {% for row in us_signal_rows %}
            <li class="signal-item {{ row.tone_class }}">{{ row.description }}</li>
            {% endfor %}
          </ul>
        </div>
        <div class="panel">
          <h3>国内侧</h3>
          <ul class="clean">
            {% for row in cn_signal_rows %}
            <li class="signal-item {{ row.tone_class }}">{{ row.description }}</li>
            {% endfor %}
          </ul>
        </div>
      </div>
    </section>

    <section>
      <h2>模型可信度摘要</h2>
      <div class="panel">
        <ul class="clean">
          {% for row in model_notes %}
          <li>{{ row }}</li>
          {% endfor %}
        </ul>
      </div>
    </section>

    <section>
      <h2>风险与数据质量</h2>
      <div class="grid-2">
        <div class="panel">
          <h3>主要风险</h3>
          <ul class="clean">
            {% for row in risk_rows %}
            <li class="signal-item {{ row.tone_class }}">{{ row.text }}</li>
            {% endfor %}
          </ul>
        </div>
        <div class="panel">
          <h3>数据状态</h3>
          <table>
            <thead><tr><th>市场</th><th>主序列</th><th>最新日期</th><th>来源</th><th>合成兜底</th></tr></thead>
            <tbody>
              {% for row in data_status_rows %}
              <tr><td>{{ row.market }}</td><td>{{ row.symbol }}</td><td>{{ row.latest_date }}</td><td>{{ row.source }}</td><td>{{ row.synthetic }}</td></tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <section>
      <h2>图表附录</h2>
      <div class="chart-grid">
        <div><img src="{{ charts.price_trend }}" alt="price trend" /><p class="chart-caption">近180个交易日中美金价走势</p></div>
        <div><img src="{{ charts.interval_band }}" alt="interval band" /><p class="chart-caption">下一交易日收益区间（P10 / 点预测 / P90）</p></div>
        <div><img src="{{ charts.cn_decomposition }}" alt="cn decomposition" /><p class="chart-caption">国内金价预测的三因子拆解：美盘 / 汇率 / 溢价</p></div>
      </div>
    </section>
  </div>
</body>
</html>
"""


FEATURE_LABELS = {
    "us_gold_usd": "美元金价",
    "us_gold_cny_per_gram": "外盘折算人民币金价",
    "usdcny_close": "美元兑人民币",
    "cn_premium": "国内溢价",
    "premium_change": "溢价变化",
    "premium_z_20": "溢价偏离度",
    "premium_vol_20": "溢价波动",
    "us_gold_return": "美元金价日收益",
    "fx_return": "汇率日变化",
    "cn_return": "国内金价日收益",
    "us_ret_1d": "美盘1日收益",
    "us_ret_5d": "美盘5日收益",
    "us_ret_20d": "美盘20日收益",
    "cn_ret_1d": "国内1日收益",
    "cn_ret_5d": "国内5日收益",
    "cn_ret_20d": "国内20日收益",
    "us_ma_20": "美盘20日均线",
    "us_ma_ratio": "美盘相对均线比值",
    "cn_ma_20": "国内20日均线",
    "cn_ma_ratio": "国内相对均线比值",
    "us_rsi_14": "美盘RSI",
    "cn_rsi_14": "国内RSI",
    "us_vol_20d": "美盘20日波动",
    "cn_vol_20d": "国内20日波动",
    "dxy": "美元指数",
    "us10y": "美债10年收益率",
    "vix": "VIX波动率",
    "spx": "标普500",
    "oil": "原油",
    "cb_gold_netbuy": "央行购金",
    "us_close": "美盘收盘价",
    "cn_close": "国内收盘价",
}

SOURCE_LABELS = {
    "lbma": "LBMA 定盘价",
    "sge": "上海黄金交易所",
    "shfe": "上期所",
    "xauusd": "Yahoo XAUUSD",
    "fx": "汇率数据",
    "macro": "宏观行情",
    "synthetic": "合成兜底数据",
}

CONTRIBUTION_LABELS = {
    "US_gold": "美盘金价",
    "FX_usdcny": "美元兑人民币",
    "CN_premium": "国内溢价",
}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _fmt_num(x: float | int | None, nd: int = 4) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return str(x)


def _fmt_price(x: float | int | None, nd: int = 2) -> str:
    return _fmt_num(x, nd=nd)


def _fmt_pct(x: float | int | None, nd: int = 2) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x) * 100:+.{nd}f}%"
    except Exception:
        return str(x)


def _fmt_prob(x: float | int | None, nd: int = 2) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x) * 100:.{nd}f}%"
    except Exception:
        return str(x)


def _fmt_ratio(x: float | int | None, nd: int = 2) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):+.{nd}f}"
    except Exception:
        return str(x)


def _fmt_position(x: float | int | None, nd: int = 1) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x) * 100:.{nd}f}%"
    except Exception:
        return str(x)


def _feature_label(feature_name: str) -> str:
    return FEATURE_LABELS.get(feature_name, feature_name)


def _source_label(source_name: str | None) -> str:
    if not source_name:
        return "N/A"
    return SOURCE_LABELS.get(source_name, source_name)


def _extract_series(prices_daily: pd.DataFrame, market: str, symbol: str) -> pd.DataFrame:
    df = prices_daily.loc[(prices_daily["market"] == market) & (prices_daily["symbol"] == symbol), ["date", "close"]].copy()
    if df.empty:
        return df
    return df.sort_values("date").drop_duplicates(subset=["date"], keep="last")


def _draw_price_trend(out_path: Path, us_series: pd.DataFrame, cn_series: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if not us_series.empty:
        p = us_series.tail(180)
        ax.plot(p["date"], p["close"], label="美盘金价", color="#c28b12", linewidth=2.2)
    if not cn_series.empty:
        p = cn_series.tail(180)
        ax.plot(p["date"], p["close"], label="国内金价", color="#2f6f74", linewidth=2.2)
    ax.set_title("中美金价近180个交易日走势")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _draw_interval_band(out_path: Path, asof: pd.Timestamp, us_pred: dict[str, Any], cn_pred: dict[str, Any]) -> None:
    def _num(x: Any, default: float = 0.0) -> float:
        value = _safe_float(x)
        return default if value is None else value

    fig, ax = plt.subplots(figsize=(10, 4))
    labels = ["美盘", "国内"]
    point = [_num(us_pred.get("next_1d_return_point")), _num(cn_pred.get("next_1d_return_point"))]
    low = [_num(us_pred.get("interval_p10")), _num(cn_pred.get("interval_p10"))]
    high = [_num(us_pred.get("interval_p90")), _num(cn_pred.get("interval_p90"))]
    x = [0, 1]
    ax.scatter(x, point, color="#1f1a14", label="点预测")
    for i in range(2):
        ax.vlines(x[i], low[i], high[i], color="#b8871b", linewidth=6, alpha=0.75)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_title(f"下一交易日收益区间（截至 {pd.to_datetime(asof).date()}）")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _draw_cn_contrib(out_path: Path, contributions: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = ["US_gold", "FX_usdcny", "CN_premium"]
    vals = [_safe_float(contributions.get(k)) or 0.0 for k in keys]
    labels = [CONTRIBUTION_LABELS.get(k, k) for k in keys]
    colors = ["#b8871b", "#2f6f74", "#9e4e3d"]
    ax.bar(labels, vals, color=colors)
    ax.axhline(0.0, color="gray", linewidth=1)
    ax.set_title("国内金价预测驱动拆解")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _series_change(series: pd.DataFrame, periods: int) -> float | None:
    if series.empty or len(series) <= periods:
        return None
    sample = series.tail(periods + 1)
    start_value = _safe_float(sample["close"].iloc[0])
    end_value = _safe_float(sample["close"].iloc[-1])
    if start_value in (None, 0.0) or end_value is None:
        return None
    return end_value / start_value - 1.0


def _series_range_position(series: pd.DataFrame, window: int = 252) -> float | None:
    if series.empty:
        return None
    sample = series.tail(min(window, len(series)))
    latest = _safe_float(sample["close"].iloc[-1])
    low = _safe_float(sample["close"].min())
    high = _safe_float(sample["close"].max())
    if latest is None or low is None or high is None or high <= low:
        return None
    return (latest - low) / (high - low)


def _build_market_snapshot(title: str, symbol: str | None, series: pd.DataFrame, pred: dict[str, Any], unit: str) -> dict[str, Any]:
    latest_close = _safe_float(pred.get("latest_close"))
    if latest_close is None and not series.empty:
        latest_close = _safe_float(series["close"].iloc[-1])

    ret_1d = _series_change(series, 1)
    ret_5d = _series_change(series, 5)
    ret_20d = _series_change(series, 20)
    range_pos = _series_range_position(series, window=252)
    point_return = _safe_float(pred.get("next_1d_return_point"))
    up_prob = _safe_float(pred.get("direction_prob_up"))
    interval_p10 = _safe_float(pred.get("interval_p10"))
    interval_p90 = _safe_float(pred.get("interval_p90"))

    if point_return is not None and up_prob is not None:
        if point_return > 0 and up_prob >= 0.55:
            commentary = "模型给出的方向较一致，偏多信号相对清晰。"
        elif point_return > 0:
            commentary = "方向偏多，但概率优势不算强，更适合等回落而不是追价。"
        elif point_return < 0 and up_prob <= 0.45:
            commentary = "短线偏弱，模型更倾向等待价格回撤后再看。"
        else:
            commentary = "点预测与概率信号分歧较大，属于方向不够统一的状态。"
    else:
        commentary = "预测信息不足，建议降低解读强度。"

    return {
        "title": title,
        "symbol": symbol or "N/A",
        "latest_close": f"{_fmt_price(latest_close)} {unit}" if latest_close is not None else "N/A",
        "point_return": _fmt_pct(point_return),
        "up_probability": _fmt_prob(up_prob),
        "interval": f"{_fmt_pct(interval_p10)} / {_fmt_pct(interval_p90)}",
        "ret_1d": _fmt_pct(ret_1d),
        "ret_5d": _fmt_pct(ret_5d),
        "ret_20d": _fmt_pct(ret_20d),
        "range_position": _fmt_position(range_pos),
        "commentary": commentary,
        "latest_close_num": latest_close,
        "point_return_num": point_return,
        "up_probability_num": up_prob,
        "interval_p10_num": interval_p10,
        "interval_p90_num": interval_p90,
        "range_position_num": range_pos,
    }


def _feature_value_text(feature_name: str, value: float | None) -> str:
    if value is None:
        return "N/A"
    if feature_name in {"spx"}:
        return _fmt_num(value, nd=0)
    if feature_name in {"dxy", "us10y", "vix", "oil", "usdcny_close", "cb_gold_netbuy"}:
        return _fmt_num(value, nd=2)
    if feature_name in {"us_gold_usd", "us_gold_cny_per_gram", "cn_premium", "us_close", "cn_close"}:
        return _fmt_price(value)
    if "ret_" in feature_name or feature_name.endswith("return") or feature_name.startswith("premium_change") or feature_name.startswith("fx_return"):
        return _fmt_pct(value)
    if "vol_" in feature_name:
        return _fmt_pct(value)
    if feature_name.endswith("_rsi_14"):
        return _fmt_num(value, nd=1)
    if feature_name.endswith("_ratio") or feature_name.endswith("_z_20"):
        return _fmt_ratio(value, nd=2)
    return _fmt_num(value, nd=4)


def _signal_tone(z_score: float) -> tuple[str, str]:
    if z_score >= 1.5:
        return "显著偏强", "positive"
    if z_score >= 0.5:
        return "略偏强", "positive"
    if z_score <= -1.5:
        return "显著偏弱", "negative"
    if z_score <= -0.5:
        return "略偏弱", "negative"
    return "接近常态", "neutral"


def _build_signal_rows(features_daily: pd.DataFrame, asof: pd.Timestamp, feature_names: list[str], top_n: int = 5) -> list[dict[str, Any]]:
    if features_daily.empty:
        return [{"description": "暂无可用特征样本。", "tone_class": "neutral"}]

    feature_wide = from_feature_long(ensure_datetime(features_daily, "date"))
    feature_wide = feature_wide.loc[feature_wide["date"] <= asof].sort_values("date")
    if feature_wide.empty:
        return [{"description": "截至报告日没有可用特征截面。", "tone_class": "neutral"}]

    rows: list[dict[str, Any]] = []
    for feature_name in feature_names:
        if feature_name not in feature_wide.columns:
            continue
        series = feature_wide[["date", feature_name]].dropna().tail(60)
        if len(series) < 20:
            continue
        latest_value = _safe_float(series[feature_name].iloc[-1])
        history = series[feature_name].iloc[:-1].astype(float)
        if latest_value is None or history.empty:
            continue
        std = float(history.std(ddof=0))
        z_score = 0.0 if std < 1e-8 else (latest_value - float(history.mean())) / std
        tone, tone_class = _signal_tone(z_score)
        rows.append(
            {
                "feature": feature_name,
                "label": _feature_label(feature_name),
                "value": _feature_value_text(feature_name, latest_value),
                "z_score": z_score,
                "tone": tone,
                "tone_class": tone_class,
                "description": f"{_feature_label(feature_name)} 当前为 {_feature_value_text(feature_name, latest_value)}，相对近60个样本 {tone}（z={z_score:+.2f}）。",
            }
        )

    if not rows:
        return [{"description": "可用于解释的特征样本不足。", "tone_class": "neutral"}]

    rows = sorted(rows, key=lambda item: abs(item["z_score"]), reverse=True)[:top_n]
    return rows


def _lag_days(asof: pd.Timestamp, latest_value: Any) -> int | None:
    if latest_value in (None, "", "N/A"):
        return None
    try:
        latest_ts = pd.to_datetime(latest_value)
    except Exception:
        return None
    return int((pd.to_datetime(asof).normalize() - latest_ts.normalize()).days)


def _data_cutoff_text(data_status: dict[str, Any], asof: pd.Timestamp) -> str:
    us_latest = data_status.get("us_latest")
    cn_latest = data_status.get("cn_latest")
    parts = []
    if us_latest:
        parts.append(f"US {pd.to_datetime(us_latest).date()}")
    if cn_latest:
        parts.append(f"CN {pd.to_datetime(cn_latest).date()}")
    if not parts:
        return str(pd.to_datetime(asof).date())
    return " / ".join(parts)


def _parse_monitoring_report(monitoring_text: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "data_quality": [],
        "drift_rows": [],
        "high_drift_count": 0,
        "medium_drift_count": 0,
        "synthetic_used": False,
        "jump_alert_markets": [],
    }
    for raw_line in monitoring_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        text = line[2:].strip()
        if "latest=" in text and "missing_rate=" in text and "source=" in text:
            prefix, _, rest = text.partition(":")
            fields = {"market_symbol": prefix.strip()}
            for part in rest.split(","):
                key, _, value = part.strip().partition("=")
                fields[key.strip()] = value.strip()
            summary["data_quality"].append(fields)
            if fields.get("recent_jump_alert", "").lower() == "true":
                summary["jump_alert_markets"].append(prefix.strip())
            continue
        if "PSI=" in text:
            feature_name, _, rest = text.partition(":")
            match = re.search(r"PSI=([0-9.]+)\s+\(([^)]+)\)", rest)
            if match:
                row = {
                    "feature": feature_name.strip(),
                    "label": _feature_label(feature_name.strip()),
                    "psi": _safe_float(match.group(1)),
                    "level": match.group(2).upper(),
                }
                summary["drift_rows"].append(row)
                if row["level"] == "HIGH":
                    summary["high_drift_count"] += 1
                elif row["level"] == "MEDIUM":
                    summary["medium_drift_count"] += 1
            continue
        if "Synthetic fallback" in text:
            summary["synthetic_used"] = True
    summary["drift_rows"] = sorted(summary["drift_rows"], key=lambda item: item.get("psi") or 0.0, reverse=True)
    return summary


def _build_driver_rows(decomposition: dict[str, Any]) -> list[dict[str, Any]]:
    contributions = decomposition.get("contributions", {})
    rows: list[dict[str, Any]] = []
    for key in ["US_gold", "FX_usdcny", "CN_premium"]:
        value = _safe_float(contributions.get(key))
        if value is None:
            value = 0.0
        if key == "US_gold":
            note = "美盘金价是国内金价的外生主驱动，正值表示外盘在抬升国内预期。"
        elif key == "FX_usdcny":
            note = "美元兑人民币走强通常会抬高人民币计价金价，走弱则形成拖累。"
        else:
            note = "国内溢价反映本地供需和定价热度，负值说明国内定价弱于外盘折算价。"
        rows.append(
            {
                "key": key,
                "label": CONTRIBUTION_LABELS.get(key, key),
                "value": _fmt_pct(value),
                "value_num": value,
                "note": note,
                "tone_class": "positive" if value > 0 else ("negative" if value < 0 else "neutral"),
            }
        )
    return rows


def _dominant_driver_text(driver_rows: list[dict[str, Any]]) -> str:
    if not driver_rows:
        return "暂无拆解结果。"
    dominant = max(driver_rows, key=lambda item: abs(item.get("value_num") or 0.0))
    value_num = dominant.get("value_num") or 0.0
    tone = "提供支撑" if value_num > 0 else ("形成拖累" if value_num < 0 else "影响中性")
    return f"{dominant['label']} 是当前主导因子，贡献 {_fmt_pct(value_num)}，对国内金价 {tone}。"


def _load_backtest_metrics(project_root: Path) -> dict[str, dict[str, Any]]:
    path = project_root / "reports" / "backtest" / "backtest_latest.md"
    if not path.exists():
        return {}

    metrics: dict[str, dict[str, Any]] = {}
    current: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            current = "us" if line.upper().startswith("## US") else ("cn" if line.upper().startswith("## CN") else None)
            if current:
                metrics[current] = {}
            continue
        if not current or not line.startswith("- "):
            continue
        if line.startswith("- observations:"):
            metrics[current]["observations"] = _safe_float(line.split(":", 1)[1].strip())
        elif line.startswith("- MAE"):
            parts = [part.strip() for part in line.split(":", 1)[1].split("/")]
            metrics[current]["main_mae"] = _safe_float(parts[-1]) if parts else None
        elif line.startswith("- Direction Acc"):
            parts = [part.strip() for part in line.split(":", 1)[1].split("/")]
            metrics[current]["main_direction_acc"] = _safe_float(parts[-1]) if parts else None
        elif line.startswith("- Probability AUC:"):
            metrics[current]["main_auc"] = _safe_float(line.split(":", 1)[1].strip())
        elif line.startswith("- Probability Brier:"):
            metrics[current]["main_brier"] = _safe_float(line.split(":", 1)[1].strip())
    return metrics


def _build_model_notes(backtest_metrics: dict[str, dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    for market_key, market_label in [("us", "美盘模型"), ("cn", "国内模型")]:
        metric = backtest_metrics.get(market_key, {})
        obs = _safe_float(metric.get("observations"))
        direction_acc = _safe_float(metric.get("main_direction_acc"))
        auc = _safe_float(metric.get("main_auc"))
        mae = _safe_float(metric.get("main_mae"))
        if obs is None and direction_acc is None and auc is None:
            notes.append(f"{market_label}：暂未读取到历史回测摘要，当前只能把模型输出当作方向参考。")
            continue
        notes.append(
            f"{market_label}：最近滚动回测样本约 {int(obs) if obs is not None else 'N/A'} 条，主模型方向正确率 {_fmt_prob(direction_acc)}，"
            f"AUC {_fmt_num(auc, nd=4)}，MAE {_fmt_num(mae, nd=4)}。"
        )
    return notes


def _build_execution_rows(cn_card: dict[str, Any]) -> list[dict[str, str]]:
    latest_close = cn_card.get("latest_close_num")
    point_return = cn_card.get("point_return_num")
    interval_p10 = cn_card.get("interval_p10_num")
    interval_p90 = cn_card.get("interval_p90_num")
    if latest_close is None:
        return [{"label": "当前收盘", "value": "N/A", "note": "当前无法推导价格区间。"}]

    mid_price = latest_close * (1.0 + (point_return or 0.0))
    bear_price = latest_close * (1.0 + (interval_p10 or 0.0))
    bull_price = latest_close * (1.0 + (interval_p90 or 0.0))
    observe_low = min(mid_price, bear_price)
    observe_high = max(mid_price, bear_price)

    if (point_return or 0.0) >= 0:
        observe_note = "若下一交易日回落到该区间，更适合分两笔布局；高于乐观情景价不建议追高。"
        avoid_line = bull_price
        avoid_note = "若价格直接突破该线，收益空间更多已经体现在模型乐观情景里。"
    else:
        observe_note = "短线偏弱时，只有回撤到该区间才考虑小仓位试探，否则优先等待。"
        avoid_line = latest_close
        avoid_note = "若价格仍高于当前收盘附近，说明性价比一般，不值得追价。"

    return [
        {"label": "当前收盘", "value": f"{_fmt_price(latest_close)} CNY/g", "note": "报告采用国内主序列最新收盘价作为定价起点。"},
        {"label": "明日中枢价", "value": f"{_fmt_price(mid_price)} CNY/g", "note": "由点预测收益换算，适合用来判断主场景。"},
        {"label": "偏弱情景价（P10）", "value": f"{_fmt_price(bear_price)} CNY/g", "note": "若市场走到这一档，说明短线偏弱情景在兑现。"},
        {"label": "偏强情景价（P90）", "value": f"{_fmt_price(bull_price)} CNY/g", "note": "若价格触及这一档，说明乐观情景基本被计入。"},
        {"label": "观察区", "value": f"{_fmt_price(observe_low)} ~ {_fmt_price(observe_high)} CNY/g", "note": observe_note},
        {"label": "不宜追高线", "value": f"{_fmt_price(avoid_line)} CNY/g", "note": avoid_note},
    ]


def _status_tuple(status: str) -> tuple[str, str]:
    if status == "满足":
        return status, "positive"
    if status == "不满足":
        return status, "negative"
    return status, "warning"


def _build_checklist_rows(asof: pd.Timestamp, cn_card: dict[str, Any], driver_rows: list[dict[str, Any]], monitoring_summary: dict[str, Any], data_status: dict[str, Any]) -> list[dict[str, str]]:
    point_return = cn_card.get("point_return_num")
    up_probability = cn_card.get("up_probability_num")
    range_position = cn_card.get("range_position_num")
    driver_map = {row["key"]: row for row in driver_rows}
    us_contrib = driver_map.get("US_gold", {}).get("value_num", 0.0)
    fx_contrib = driver_map.get("FX_usdcny", {}).get("value_num", 0.0)
    premium_contrib = driver_map.get("CN_premium", {}).get("value_num", 0.0)
    cn_lag = _lag_days(asof, data_status.get("cn_latest"))
    high_drift_count = int(monitoring_summary.get("high_drift_count", 0))

    items = [
        ("短线方向", "满足" if (point_return or 0.0) > 0 else ("留意" if (point_return or 0.0) == 0 else "不满足"), f"国内点预测为 {_fmt_pct(point_return)}。"),
        ("上涨概率优势", "满足" if (up_probability or 0.0) >= 0.55 else ("留意" if (up_probability or 0.0) >= 0.48 else "不满足"), f"国内上涨概率为 {_fmt_prob(up_probability)}。"),
        ("外盘共振", "满足" if us_contrib > 0 else ("留意" if us_contrib == 0 else "不满足"), f"美盘贡献为 {_fmt_pct(us_contrib)}。"),
        ("汇率顺风", "满足" if fx_contrib > 0 else ("留意" if fx_contrib == 0 else "不满足"), f"汇率贡献为 {_fmt_pct(fx_contrib)}。"),
        ("国内溢价拖累", "满足" if premium_contrib >= 0 else ("留意" if premium_contrib >= -0.002 else "不满足"), f"溢价贡献为 {_fmt_pct(premium_contrib)}。"),
        ("价格位置", "满足" if range_position is not None and range_position <= 0.70 else ("留意" if range_position is not None and range_position <= 0.85 else "不满足"), f"国内价格位于近1年区间的 {_fmt_position(range_position)}。"),
        ("数据时效", "满足" if cn_lag is not None and cn_lag <= 3 else ("留意" if cn_lag is not None and cn_lag <= 5 else "不满足"), f"国内最新时间戳相对报告日滞后 {cn_lag if cn_lag is not None else 'N/A'} 天。"),
        ("监控漂移", "不满足" if monitoring_summary.get("synthetic_used") else ("满足" if high_drift_count <= 3 else ("留意" if high_drift_count <= 8 else "不满足")), f"当前 HIGH 级别漂移项共 {high_drift_count} 个。"),
    ]
    rows = []
    for label, status, detail in items:
        status, status_class = _status_tuple(status)
        rows.append({"label": label, "status": status, "status_class": status_class, "detail": detail})
    return rows


def _build_core_view(asof: pd.Timestamp, cn_card: dict[str, Any], driver_rows: list[dict[str, Any]], monitoring_summary: dict[str, Any], data_status: dict[str, Any], backtest_metrics: dict[str, dict[str, Any]]) -> dict[str, str]:
    point_return = cn_card.get("point_return_num") or 0.0
    up_probability = cn_card.get("up_probability_num") or 0.0
    range_position = cn_card.get("range_position_num")
    cn_lag = _lag_days(asof, data_status.get("cn_latest"))
    dominant_driver = _dominant_driver_text(driver_rows)
    high_drift_count = int(monitoring_summary.get("high_drift_count", 0))
    cn_direction_acc = _safe_float(backtest_metrics.get("cn", {}).get("main_direction_acc"))

    if point_return >= 0.005 and up_probability >= 0.58:
        action_tag, tone_class = "偏多，可分批买入", "positive"
    elif point_return >= 0 and up_probability >= 0.50:
        action_tag, tone_class = "中性偏多，回落再买", "warning"
    elif point_return <= -0.005 and up_probability <= 0.42:
        action_tag, tone_class = "短线偏弱，等待更优价格", "negative"
    else:
        action_tag, tone_class = "信号分歧，控制节奏", "warning"
    if range_position is not None and range_position >= 0.85 and point_return >= 0:
        action_tag, tone_class = "高位偏贵，勿追价", "warning"

    confidence_score = 0
    if up_probability >= 0.55:
        confidence_score += 1
    elif up_probability <= 0.45:
        confidence_score -= 1
    if abs(point_return) >= 0.003:
        confidence_score += 1
    if cn_direction_acc is not None and cn_direction_acc >= 0.65:
        confidence_score += 1
    elif cn_direction_acc is not None and cn_direction_acc < 0.55:
        confidence_score -= 1
    if high_drift_count >= 6:
        confidence_score -= 1
    if cn_lag is not None and cn_lag > 3:
        confidence_score -= 1
    confidence_label = "中高" if confidence_score >= 2 else ("中等" if confidence_score >= 0 else "偏低")

    summary = f"模型对下一交易日国内金价的判断为 {action_tag}：点预测 {_fmt_pct(point_return)}，上涨概率 {_fmt_prob(up_probability)}。{dominant_driver}"
    action_line = (
        "优先等价格回落到模型观察区再分批，若直接高开到乐观情景价附近，不建议追高。"
        if point_return >= 0
        else "短线更适合等回撤确认，不建议在高于当前收盘附近的价格位置抢跑。"
    )
    if cn_lag is None:
        freshness_note = "未读取到国内主序列最新时间戳。"
    elif cn_lag <= 2 and pd.to_datetime(asof).weekday() >= 5:
        freshness_note = f"国内数据最新至 {pd.to_datetime(data_status.get('cn_latest')).date()}，相对 {pd.to_datetime(asof).date()} 的滞后主要由周末造成。"
    else:
        freshness_note = f"国内数据最新至 {pd.to_datetime(data_status.get('cn_latest')).date()}，相对报告日滞后 {cn_lag} 天。"

    if monitoring_summary.get("synthetic_used"):
        constraint_line = "存在合成数据兜底，需显著下调结论强度。"
    elif high_drift_count >= 6:
        constraint_line = f"监控中出现 {high_drift_count} 个 HIGH 漂移项，说明当前市场状态和历史训练分布偏离较大。"
    else:
        constraint_line = "暂无合成兜底，数据时效可接受，但仍需结合区间风险控制仓位。"

    execution_note = f"这份报告先给一句话结论，再给价格区间和检查清单。若你只看一个结论：{action_line}"
    return {
        "summary": summary,
        "action_tag": action_tag,
        "tone_class": tone_class,
        "confidence_label": confidence_label,
        "action_line": action_line,
        "execution_note": execution_note,
        "dominant_driver": dominant_driver,
        "freshness_note": freshness_note,
        "constraint_line": constraint_line,
    }


def _build_risk_rows(asof: pd.Timestamp, data_status: dict[str, Any], monitoring_summary: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    us_lag = _lag_days(asof, data_status.get("us_latest"))
    cn_lag = _lag_days(asof, data_status.get("cn_latest"))
    if cn_lag is not None:
        rows.append({"text": f"国内主序列最新日期为 {pd.to_datetime(data_status.get('cn_latest')).date()}，相对报告日滞后 {cn_lag} 天。", "tone_class": "warning" if cn_lag <= 3 else "negative"})
    if us_lag is not None:
        rows.append({"text": f"美盘主序列最新日期为 {pd.to_datetime(data_status.get('us_latest')).date()}，相对报告日滞后 {us_lag} 天。", "tone_class": "warning" if us_lag <= 3 else "negative"})
    if monitoring_summary.get("synthetic_used"):
        rows.append({"text": "监控报告提示至少一个主序列启用了 synthetic fallback，这会直接降低结论可信度。", "tone_class": "negative"})
    if monitoring_summary.get("jump_alert_markets"):
        rows.append({"text": f"近期跳变告警触发于：{', '.join(monitoring_summary['jump_alert_markets'])}，需警惕异常波动导致的模型失真。", "tone_class": "negative"})
    for drift_row in monitoring_summary.get("drift_rows", [])[:4]:
        rows.append({"text": f"{drift_row['label']} 的 PSI 为 {_fmt_num(drift_row.get('psi'), nd=4)}（{drift_row['level']}），说明当前分布与历史样本偏离明显。", "tone_class": "warning" if drift_row["level"] == "HIGH" else "neutral"})
    if not rows:
        rows.append({"text": "未发现明显的时效、漂移或异常跳变风险。", "tone_class": "positive"})
    return rows


def _build_data_status_rows(data_status: dict[str, Any], active_symbols: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for mk, market in [("us", "美盘"), ("cn", "国内")]:
        latest = data_status.get(f"{mk}_latest")
        rows.append(
            {
                "market": market,
                "symbol": active_symbols.get(mk, {}).get("symbol", "N/A"),
                "latest_date": str(pd.to_datetime(latest).date()) if latest else "N/A",
                "source": _source_label(data_status.get(mk, {}).get("chosen_source")),
                "synthetic": "是" if bool(data_status.get(mk, {}).get("used_synthetic", False)) else "否",
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
    report_title = f"黄金购买参考日报（中文版）- {report_date}"
    daily_dir = project_root / "reports" / "daily"
    assets_dir = daily_dir / "assets"
    daily_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    us_symbol = active_symbols.get("us", {}).get("symbol")
    cn_symbol = active_symbols.get("cn", {}).get("symbol")
    us_series = _extract_series(prices_daily, "US", us_symbol) if us_symbol else pd.DataFrame()
    cn_series = _extract_series(prices_daily, "CN", cn_symbol) if cn_symbol else pd.DataFrame()
    us_pred = market_results.get("us", {}).get("prediction", {})
    cn_pred = market_results.get("cn", {}).get("prediction", {})

    price_chart = assets_dir / f"{report_date}_price_trend.png"
    interval_chart = assets_dir / f"{report_date}_interval_band.png"
    decomp_chart = assets_dir / f"{report_date}_cn_decomp.png"
    _draw_price_trend(price_chart, us_series, cn_series)
    _draw_interval_band(interval_chart, asof, us_pred, cn_pred)
    _draw_cn_contrib(decomp_chart, decomposition.get("contributions", {}))

    monitoring_text = monitoring_report_path.read_text(encoding="utf-8") if monitoring_report_path.exists() else ""
    monitoring_summary = _parse_monitoring_report(monitoring_text)
    backtest_metrics = _load_backtest_metrics(project_root)
    us_card = _build_market_snapshot("美盘金价", us_symbol, us_series, us_pred, "USD/oz")
    cn_card = _build_market_snapshot("国内金价", cn_symbol, cn_series, cn_pred, "CNY/g")
    driver_rows = _build_driver_rows(decomposition)
    execution_rows = _build_execution_rows(cn_card)
    checklist_rows = _build_checklist_rows(asof, cn_card, driver_rows, monitoring_summary, data_status)
    core_view = _build_core_view(asof, cn_card, driver_rows, monitoring_summary, data_status, backtest_metrics)
    risk_rows = _build_risk_rows(asof, data_status, monitoring_summary)
    data_rows = _build_data_status_rows(data_status=data_status, active_symbols=active_symbols)
    model_notes = _build_model_notes(backtest_metrics)
    us_signal_rows = _build_signal_rows(features_daily, asof=asof, feature_names=["us_gold_usd", "us_ret_5d", "us_ret_20d", "us_rsi_14", "dxy", "us10y", "vix", "spx", "oil"])
    cn_signal_rows = _build_signal_rows(features_daily, asof=asof, feature_names=["cn_close", "cn_ret_5d", "cn_ret_20d", "cn_premium", "premium_z_20", "usdcny_close", "fx_return", "cb_gold_netbuy"])

    env = Environment(autoescape=True, trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(HTML_TEMPLATE)
    html = template.render(
        report_title=report_title,
        report_date=report_date,
        data_cutoff_text=_data_cutoff_text(data_status, asof),
        core_view=core_view,
        execution_rows=execution_rows,
        checklist_rows=checklist_rows,
        market_cards=[us_card, cn_card],
        driver_rows=driver_rows,
        us_signal_rows=us_signal_rows,
        cn_signal_rows=cn_signal_rows,
        model_notes=model_notes,
        risk_rows=risk_rows,
        data_status_rows=data_rows,
        charts={
            "price_trend": str(price_chart.name if price_chart.parent == daily_dir else f"assets/{price_chart.name}"),
            "interval_band": str(interval_chart.name if interval_chart.parent == daily_dir else f"assets/{interval_chart.name}"),
            "cn_decomposition": str(decomp_chart.name if decomp_chart.parent == daily_dir else f"assets/{decomp_chart.name}"),
        },
    )

    html_path = daily_dir / f"{report_date}_gold_report.html"
    md_path = daily_dir / f"{report_date}_gold_report.md"
    html_path.write_text(html, encoding="utf-8")

    md_lines = [f"# {report_title}", "", "## 一句话核心结论", f"- {core_view['summary']}", f"- 执行建议：{core_view['action_line']}", f"- 结论置信度：{core_view['confidence_label']}", f"- 数据时效：{core_view['freshness_note']}", "", "## 今日操作计划"]
    for row in execution_rows:
        md_lines.append(f"- {row['label']}：{row['value']}。{row['note']}")
    md_lines.extend(["", "## 操作检查清单"])
    for row in checklist_rows:
        md_lines.append(f"- [{row['status']}] {row['label']}：{row['detail']}")
    md_lines.extend(["", "## 市场状态概览", "### 美盘"])
    for line in [f"最新价格：{us_card['latest_close']}", f"模型点预测：{us_card['point_return']}", f"上涨概率：{us_card['up_probability']}", f"P10 / P90：{us_card['interval']}", f"近1日 / 5日 / 20日：{us_card['ret_1d']} / {us_card['ret_5d']} / {us_card['ret_20d']}", f"近1年区间位置：{us_card['range_position']}", us_card["commentary"]]:
        md_lines.append(f"- {line}")
    md_lines.extend(["", "### 国内"])
    for line in [f"最新价格：{cn_card['latest_close']}", f"模型点预测：{cn_card['point_return']}", f"上涨概率：{cn_card['up_probability']}", f"P10 / P90：{cn_card['interval']}", f"近1日 / 5日 / 20日：{cn_card['ret_1d']} / {cn_card['ret_5d']} / {cn_card['ret_20d']}", f"近1年区间位置：{cn_card['range_position']}", cn_card["commentary"]]:
        md_lines.append(f"- {line}")
    md_lines.extend(["", "## 驱动拆解"])
    for row in driver_rows:
        md_lines.append(f"- {row['label']}：{row['value']}。{row['note']}")
    md_lines.extend(["", "## 重点信号", "### 美盘侧"])
    for row in us_signal_rows:
        md_lines.append(f"- {row['description']}")
    md_lines.extend(["", "### 国内侧"])
    for row in cn_signal_rows:
        md_lines.append(f"- {row['description']}")
    md_lines.extend(["", "## 模型可信度摘要"])
    for row in model_notes:
        md_lines.append(f"- {row}")
    md_lines.extend(["", "## 风险与数据质量"])
    for row in risk_rows:
        md_lines.append(f"- {row['text']}")
    md_lines.extend(["", "## 数据来源"])
    for row in data_rows:
        md_lines.append(f"- {row['market']}：主序列 {row['symbol']}，最新日期 {row['latest_date']}，来源 {row['source']}，合成兜底 {row['synthetic']}。")
    md_lines.extend(["", "> 注：以上为基于历史数据、滚动回测与下一交易日收益预测生成的量化简报，不构成个性化投资建议。"])
    md_path.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")
    return html_path, md_path
