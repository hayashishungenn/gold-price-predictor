from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline_io import ensure_datetime, from_feature_long


LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    market: str
    symbol: str
    data: pd.DataFrame
    feature_cols: list[str]


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def _direction_accuracy(y_true_ret: np.ndarray, y_pred_ret: np.ndarray) -> float:
    true_dir = (y_true_ret > 0).astype(int)
    pred_dir = (y_pred_ret > 0).astype(int)
    return float((true_dir == pred_dir).mean())


def _prepare_market_dataset(
    prices_daily: pd.DataFrame,
    features_daily: pd.DataFrame,
    market: str,
    symbol: str,
) -> DatasetBundle:
    prices = ensure_datetime(prices_daily, "date")
    features = ensure_datetime(features_daily, "date")
    price_series = (
        prices.loc[(prices["market"] == market) & (prices["symbol"] == symbol), ["date", "close"]]
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
    )
    feat_wide = from_feature_long(features) if not features.empty else pd.DataFrame(columns=["date"])
    merged = price_series.merge(feat_wide, on="date", how="left").sort_values("date")
    merged["target_next_1d_return"] = merged["close"].pct_change().shift(-1)
    merged["target_up"] = (merged["target_next_1d_return"] > 0).astype(int)
    merged["target_vol"] = merged["close"].pct_change().rolling(20, min_periods=20).std().shift(-1)
    merged = merged.dropna(subset=["target_next_1d_return"]).reset_index(drop=True)

    non_feature_cols = {"date", "close", "target_next_1d_return", "target_up", "target_vol"}
    feature_cols = [c for c in merged.columns if c not in non_feature_cols]
    if not feature_cols:
        merged["bias"] = 1.0
        feature_cols = ["bias"]
    merged[feature_cols] = merged[feature_cols].replace([np.inf, -np.inf], np.nan)
    merged[feature_cols] = merged[feature_cols].ffill().fillna(0.0)
    return DatasetBundle(market=market, symbol=symbol, data=merged, feature_cols=feature_cols)


def _build_ridge_model() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=42))])


def _build_main_return_model() -> Any:
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression",
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    except Exception:
        return GradientBoostingRegressor(random_state=42)


def _build_main_direction_model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=42)


def _build_quantile_model(alpha: float) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=42)


def _rolling_backtest(bundle: DatasetBundle, min_train_size: int = 252) -> pd.DataFrame:
    df = bundle.data.copy()
    n = len(df)
    if n < max(min_train_size + 20, 80):
        return pd.DataFrame()

    step = max(1, int((n - min_train_size) / 120))
    rows: list[dict[str, Any]] = []

    for i in range(min_train_size, n, step):
        train = df.iloc[:i]
        test = df.iloc[i : i + 1]
        if test.empty:
            continue
        x_train = train[bundle.feature_cols].values
        y_train_ret = train["target_next_1d_return"].values
        y_train_up = train["target_up"].values
        x_test = test[bundle.feature_cols].values
        y_test_ret = test["target_next_1d_return"].values
        y_test_up = test["target_up"].values

        ridge = _build_ridge_model()
        ridge.fit(x_train, y_train_ret)
        ridge_pred = float(ridge.predict(x_test)[0])

        main_reg = _build_main_return_model()
        main_reg.fit(x_train, y_train_ret)
        main_pred = float(main_reg.predict(x_test)[0])

        cls = _build_main_direction_model()
        cls.fit(x_train, y_train_up)
        try:
            cls_prob = float(cls.predict_proba(x_test)[0, 1])
        except Exception:
            cls_prob = float(cls.predict(x_test)[0])

        rows.append(
            {
                "date": test["date"].iloc[0],
                "y_true_return": float(y_test_ret[0]),
                "y_true_up": int(y_test_up[0]),
                "rw_pred_return": 0.0,
                "ridge_pred_return": ridge_pred,
                "main_pred_return": main_pred,
                "main_prob_up": cls_prob,
            }
        )
    return pd.DataFrame(rows)


def _summarize_backtest(backtest_df: pd.DataFrame) -> dict[str, Any]:
    if backtest_df.empty:
        return {
            "n_obs": 0,
            "rw_mae": None,
            "ridge_mae": None,
            "main_mae": None,
            "rw_rmse": None,
            "ridge_rmse": None,
            "main_rmse": None,
            "rw_direction_acc": None,
            "ridge_direction_acc": None,
            "main_direction_acc": None,
            "main_auc": None,
            "main_brier": None,
        }
    y = backtest_df["y_true_return"].values
    rw = backtest_df["rw_pred_return"].values
    rg = backtest_df["ridge_pred_return"].values
    mn = backtest_df["main_pred_return"].values
    y_up = backtest_df["y_true_up"].values
    prob = backtest_df["main_prob_up"].values

    return {
        "n_obs": int(len(backtest_df)),
        "rw_mae": float(mean_absolute_error(y, rw)),
        "ridge_mae": float(mean_absolute_error(y, rg)),
        "main_mae": float(mean_absolute_error(y, mn)),
        "rw_rmse": float(np.sqrt(mean_squared_error(y, rw))),
        "ridge_rmse": float(np.sqrt(mean_squared_error(y, rg))),
        "main_rmse": float(np.sqrt(mean_squared_error(y, mn))),
        "rw_direction_acc": _direction_accuracy(y, rw),
        "ridge_direction_acc": _direction_accuracy(y, rg),
        "main_direction_acc": _direction_accuracy(y, mn),
        "main_auc": _safe_auc(y_up, prob),
        "main_brier": float(brier_score_loss(y_up, np.clip(prob, 1e-6, 1 - 1e-6))),
    }


def _format_metrics_md(title: str, metrics: dict[str, Any]) -> str:
    lines = [f"## {title}", "", f"- observations: {metrics.get('n_obs')}"]
    lines.append(f"- MAE (RW/Ridge/Main): {metrics.get('rw_mae')} / {metrics.get('ridge_mae')} / {metrics.get('main_mae')}")
    lines.append(f"- RMSE (RW/Ridge/Main): {metrics.get('rw_rmse')} / {metrics.get('ridge_rmse')} / {metrics.get('main_rmse')}")
    lines.append(
        "- Direction Acc (RW/Ridge/Main): "
        f"{metrics.get('rw_direction_acc')} / {metrics.get('ridge_direction_acc')} / {metrics.get('main_direction_acc')}"
    )
    lines.append(f"- Probability AUC: {metrics.get('main_auc')}")
    lines.append(f"- Probability Brier: {metrics.get('main_brier')}")
    return "\n".join(lines)


def run_backtest_and_train_market(
    project_root: Path,
    prices_daily: pd.DataFrame,
    features_daily: pd.DataFrame,
    market: str,
    symbol: str,
    asof: pd.Timestamp,
) -> dict[str, Any]:
    bundle = _prepare_market_dataset(prices_daily, features_daily, market=market, symbol=symbol)
    df = bundle.data.copy()
    if df.empty:
        raise ValueError(f"No modeling data for {market}/{symbol}")

    backtest_df = _rolling_backtest(bundle=bundle)
    metrics = _summarize_backtest(backtest_df)

    x = df[bundle.feature_cols].values
    y_ret = df["target_next_1d_return"].values
    y_up = df["target_up"].values
    y_vol = df["target_vol"].fillna(df["target_vol"].median()).values

    ridge = _build_ridge_model()
    ridge.fit(x, y_ret)

    main_return = _build_main_return_model()
    main_return.fit(x, y_ret)

    main_direction = _build_main_direction_model()
    main_direction.fit(x, y_up)
    calibrator = CalibratedClassifierCV(estimator=_build_main_direction_model(), method="isotonic", cv=3)
    calibrator.fit(x, y_up)

    vol_model = _build_ridge_model()
    vol_model.fit(x, y_vol)

    q10 = _build_quantile_model(alpha=0.1)
    q50 = _build_quantile_model(alpha=0.5)
    q90 = _build_quantile_model(alpha=0.9)
    q10.fit(x, y_ret)
    q50.fit(x, y_ret)
    q90.fit(x, y_ret)

    model_dir = project_root / "models"
    baseline_dir = model_dir / "baseline"
    main_dir = model_dir / "main"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    main_dir.mkdir(parents=True, exist_ok=True)

    market_key = market.lower()
    joblib.dump(ridge, baseline_dir / f"{market_key}_ridge.pkl")
    joblib.dump(main_return, main_dir / f"{market_key}_main_return_model.pkl")
    joblib.dump(main_direction, main_dir / f"{market_key}_main_direction_model.pkl")
    joblib.dump(calibrator, main_dir / f"{market_key}_calibrator.pkl")
    joblib.dump(vol_model, main_dir / f"{market_key}_vol_model.pkl")
    joblib.dump(q10, main_dir / f"{market_key}_q10_model.pkl")
    joblib.dump(q50, main_dir / f"{market_key}_q50_model.pkl")
    joblib.dump(q90, main_dir / f"{market_key}_q90_model.pkl")
    joblib.dump(bundle.feature_cols, main_dir / f"{market_key}_feature_cols.pkl")

    latest = df.loc[df["date"] <= asof].tail(1)
    if latest.empty:
        latest = df.tail(1)
    x_latest = latest[bundle.feature_cols].values
    point_pred = float(main_return.predict(x_latest)[0])
    prob_up = float(calibrator.predict_proba(x_latest)[0, 1])
    q10_pred = float(q10.predict(x_latest)[0])
    q50_pred = float(q50.predict(x_latest)[0])
    q90_pred = float(q90.predict(x_latest)[0])
    vol_pred = float(max(vol_model.predict(x_latest)[0], 1e-6))

    latest_close = float(latest["close"].iloc[0])
    prediction = {
        "market": market,
        "symbol": symbol,
        "asof": str(pd.to_datetime(latest["date"].iloc[0]).date()),
        "latest_close": latest_close,
        "next_1d_return_point": point_pred,
        "direction_prob_up": prob_up,
        "interval_p10": q10_pred,
        "interval_p50": q50_pred,
        "interval_p90": q90_pred,
        "vol_forecast": vol_pred,
    }
    return {"prediction": prediction, "metrics": metrics, "backtest_df": backtest_df}


def train_cn_decomposition(
    project_root: Path,
    prices_daily: pd.DataFrame,
    features_daily: pd.DataFrame,
    cn_symbol: str,
    asof: pd.Timestamp,
) -> dict[str, Any]:
    features = from_feature_long(ensure_datetime(features_daily, "date"))
    required_cols = ["us_gold_return", "fx_return", "premium_change", "cn_premium", "cn_return"]
    for col in required_cols:
        if col not in features.columns:
            return {"route": "decomposition", "available": False, "reason": f"missing feature: {col}"}

    data = features.sort_values("date").copy()
    data["target_us_ret"] = data["us_gold_return"].shift(-1)
    data["target_fx_ret"] = data["fx_return"].shift(-1)
    data["target_premium_chg"] = data["premium_change"].shift(-1)
    data = data.dropna(subset=["target_us_ret", "target_fx_ret", "target_premium_chg"])

    if data.empty:
        return {"route": "decomposition", "available": False, "reason": "insufficient cross-market history"}

    non_feature_cols = {"date", "target_us_ret", "target_fx_ret", "target_premium_chg"}
    feature_cols = [c for c in data.columns if c not in non_feature_cols]
    data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    x = data[feature_cols].values
    us_model = _build_ridge_model()
    fx_model = _build_ridge_model()
    prem_model = _build_ridge_model()
    us_model.fit(x, data["target_us_ret"].values)
    fx_model.fit(x, data["target_fx_ret"].values)
    prem_model.fit(x, data["target_premium_chg"].values)

    main_dir = project_root / "models" / "main"
    main_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(us_model, main_dir / "cn_component_us_return_model.pkl")
    joblib.dump(fx_model, main_dir / "cn_component_fx_return_model.pkl")
    joblib.dump(prem_model, main_dir / "cn_component_premium_change_model.pkl")
    joblib.dump(feature_cols, main_dir / "cn_component_feature_cols.pkl")

    latest = data.loc[data["date"] <= asof].tail(1)
    if latest.empty:
        latest = data.tail(1)
    x_latest = latest[feature_cols].values
    pred_us = float(us_model.predict(x_latest)[0])
    pred_fx = float(fx_model.predict(x_latest)[0])
    pred_premium = float(prem_model.predict(x_latest)[0])

    cn_close_series = (
        prices_daily.loc[(prices_daily["market"] == "CN") & (prices_daily["symbol"] == cn_symbol), ["date", "close"]]
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
    )
    cn_close = float(cn_close_series.loc[cn_close_series["date"] <= asof, "close"].tail(1).iloc[0]) if not cn_close_series.empty else 1.0
    premium_return_component = pred_premium / max(cn_close, 1e-6)
    cn_return_pred = pred_us + pred_fx + premium_return_component

    payload = {
        "route": "decomposition",
        "available": True,
        "asof": str(pd.to_datetime(latest["date"].iloc[0]).date()),
        "pred_us_return": pred_us,
        "pred_fx_return": pred_fx,
        "pred_premium_change": pred_premium,
        "premium_return_component": premium_return_component,
        "pred_cn_return_from_components": cn_return_pred,
        "contributions": {
            "US_gold": pred_us,
            "FX_usdcny": pred_fx,
            "CN_premium": premium_return_component,
        },
    }
    return payload


def write_backtest_report(project_root: Path, market_results: dict[str, dict[str, Any]]) -> None:
    lines = ["# Backtest Latest", ""]
    for market_key, result in market_results.items():
        title = f"{market_key.upper()} ({result['prediction']['symbol']})"
        lines.append(_format_metrics_md(title, result["metrics"]))
        lines.append("")
    content = "\n".join(lines).strip() + "\n"
    report_md = project_root / "reports" / "backtest" / "backtest_latest.md"
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(content, encoding="utf-8")

    report_html = project_root / "reports" / "backtest" / "backtest_latest.html"
    report_html.write_text(
        "<html><body><pre>" + content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + "</pre></body></html>",
        encoding="utf-8",
    )


def write_training_artifacts(
    project_root: Path,
    market_results: dict[str, dict[str, Any]],
    decomposition_result: dict[str, Any],
) -> None:
    payload = {
        "markets": {k: v["prediction"] for k, v in market_results.items()},
        "decomposition": decomposition_result,
    }
    artifact_path = project_root / "models" / "main" / "latest_predictions.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def train_and_backtest(
    project_root: Path,
    prices_daily: pd.DataFrame,
    features_daily: pd.DataFrame,
    active_symbols: dict[str, Any],
    asof: pd.Timestamp,
) -> dict[str, Any]:
    us_symbol = active_symbols.get("us", {}).get("symbol")
    cn_symbol = active_symbols.get("cn", {}).get("symbol")
    if not us_symbol or not cn_symbol:
        raise ValueError("Active symbols missing for US/CN market.")

    market_results = {
        "us": run_backtest_and_train_market(
            project_root=project_root,
            prices_daily=prices_daily,
            features_daily=features_daily,
            market="US",
            symbol=us_symbol,
            asof=asof,
        ),
        "cn": run_backtest_and_train_market(
            project_root=project_root,
            prices_daily=prices_daily,
            features_daily=features_daily,
            market="CN",
            symbol=cn_symbol,
            asof=asof,
        ),
    }
    decomposition = train_cn_decomposition(
        project_root=project_root,
        prices_daily=prices_daily,
        features_daily=features_daily,
        cn_symbol=cn_symbol,
        asof=asof,
    )
    write_backtest_report(project_root=project_root, market_results=market_results)
    write_training_artifacts(project_root=project_root, market_results=market_results, decomposition_result=decomposition)

    return {"market_results": market_results, "decomposition": decomposition}


def _load_model_or_none(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def infer_from_saved_models(
    project_root: Path,
    prices_daily: pd.DataFrame,
    features_daily: pd.DataFrame,
    active_symbols: dict[str, Any],
    asof: pd.Timestamp,
) -> dict[str, Any]:
    output: dict[str, Any] = {"market_results": {}, "decomposition": {"available": False}}
    for mk, market in [("us", "US"), ("cn", "CN")]:
        symbol = active_symbols.get(mk, {}).get("symbol")
        if not symbol:
            continue
        bundle = _prepare_market_dataset(prices_daily, features_daily, market=market, symbol=symbol)
        df = bundle.data
        if df.empty:
            continue
        latest = df.loc[df["date"] <= asof].tail(1)
        if latest.empty:
            latest = df.tail(1)

        model_dir = project_root / "models" / "main"
        market_key = mk
        ret_model = _load_model_or_none(model_dir / f"{market_key}_main_return_model.pkl")
        cal = _load_model_or_none(model_dir / f"{market_key}_calibrator.pkl")
        q10 = _load_model_or_none(model_dir / f"{market_key}_q10_model.pkl")
        q50 = _load_model_or_none(model_dir / f"{market_key}_q50_model.pkl")
        q90 = _load_model_or_none(model_dir / f"{market_key}_q90_model.pkl")
        vol_model = _load_model_or_none(model_dir / f"{market_key}_vol_model.pkl")
        expected_cols = _load_model_or_none(model_dir / f"{market_key}_feature_cols.pkl")
        if not isinstance(expected_cols, list) or not expected_cols:
            expected_cols = bundle.feature_cols

        if ret_model is None or cal is None:
            continue
        latest_aligned = latest.copy()
        for col in expected_cols:
            if col not in latest_aligned.columns:
                latest_aligned[col] = 0.0
        latest_aligned[expected_cols] = latest_aligned[expected_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x_latest = latest_aligned[expected_cols].values

        point_pred = float(ret_model.predict(x_latest)[0])
        prob_up = float(cal.predict_proba(x_latest)[0, 1])
        if q10 and q50 and q90:
            p10 = float(q10.predict(x_latest)[0])
            p50 = float(q50.predict(x_latest)[0])
            p90 = float(q90.predict(x_latest)[0])
        else:
            sigma = float(vol_model.predict(x_latest)[0]) if vol_model else 0.01
            p10 = point_pred - 1.2816 * sigma
            p50 = point_pred
            p90 = point_pred + 1.2816 * sigma
        vol_pred = float(vol_model.predict(x_latest)[0]) if vol_model else float(abs(point_pred))

        output["market_results"][mk] = {
            "prediction": {
                "market": market,
                "symbol": symbol,
                "asof": str(pd.to_datetime(latest["date"].iloc[0]).date()),
                "latest_close": float(latest["close"].iloc[0]),
                "next_1d_return_point": point_pred,
                "direction_prob_up": prob_up,
                "interval_p10": p10,
                "interval_p50": p50,
                "interval_p90": p90,
                "vol_forecast": vol_pred,
            },
            "metrics": {},
            "backtest_df": pd.DataFrame(),
        }

    cn_symbol = active_symbols.get("cn", {}).get("symbol")
    if cn_symbol:
        output["decomposition"] = train_cn_decomposition(
            project_root=project_root,
            prices_daily=prices_daily,
            features_daily=features_daily,
            cn_symbol=cn_symbol,
            asof=asof,
        )
    return output
