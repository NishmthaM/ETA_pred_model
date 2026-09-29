"""Training / evaluation CLI.

    python -m src.training --evaluate-only   # default: keep saved model, (re)compute metrics
    python -m src.training --retrain         # retrain XGBoost and overwrite models/*
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import (DATA_FILE, FEATURE_FILE, IMPORTANCE_FILE, METRICS_FILE, MODEL_FILE, SEED,
                     TEST_FRACTION, VALID_FRACTION)
from .data_loader import load_journeys
from .feature_engineering import FEATURE_COLUMNS, add_features, model_matrix
from .model_service import ModelService


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Featured rows that have a label (the final station has no 'next' station)."""
    data = add_features(df)
    data = data.dropna(subset=["target_delay_min"])
    data["target_change"] = data["target_delay_min"] - data["current_delay_min"]
    return data.reset_index(drop=True)


def split_by_journey(data: pd.DataFrame):
    """Chronological split BY JOURNEY: oldest 80 % train (last 15 % of those = early-stopping
    validation), newest 20 % = untouched test set. Whole journeys never straddle a split."""
    ids = data.sort_values(["journey_date", "journey_pos"])["journey_id"].drop_duplicates().tolist()
    cut = int(len(ids) * (1 - TEST_FRACTION))
    train_ids, test_ids = ids[:cut], ids[cut:]
    vcut = int(len(train_ids) * (1 - VALID_FRACTION))
    fit_ids, val_ids = train_ids[:vcut], train_ids[vcut:]
    pick = lambda s: data[data["journey_id"].isin(set(s))]
    return pick(fit_ids), pick(val_ids), pick(test_ids), (fit_ids, val_ids, test_ids)


def _scores(actual, pred) -> dict:
    return {"mae": float(mean_absolute_error(actual, pred)),
            "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
            "r2": float(r2_score(actual, pred)),
            "within_2min_pct": float(np.mean(np.abs(actual - pred) <= 2) * 100)}


def evaluate(model: ModelService, data: pd.DataFrame, test: pd.DataFrame, ids, meta: dict) -> dict:
    change = model.predict_change(test)
    pred = model.to_next_delay(test["current_delay_min"].values, change)
    act = test["target_delay_min"].values
    base = test["current_delay_min"].values                       # persistence baseline
    fit_ids, val_ids, test_ids = ids
    monthly = test.groupby("month").size().to_dict()
    return {
        "data_source": "SYNTHETIC",
        "target": "change in delay (target_delay_min - current_delay_min); "
                  "next delay = current + predicted change, floored at 0",
        "n_journeys_total": int(data["journey_id"].nunique()),
        "n_samples_labelled": int(len(data)),
        "n_train_journeys": len(fit_ids), "n_val_journeys": len(val_ids), "n_test_journeys": len(test_ids),
        "test_date_range": [str(test["journey_date"].min().date()), str(test["journey_date"].max().date())],
        "test_rows_by_month": {int(k): int(v) for k, v in monthly.items()},
        "n_test_samples": int(len(test)),
        "xgboost": _scores(act, pred),
        "persistence_baseline": _scores(act, base),
        "xgboost_change_only": _scores(test["target_change"].values, change),  # R2 on the hard part
        **meta,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true", help="retrain and overwrite saved model")
    ap.add_argument("--evaluate-only", action="store_true", help="(default) keep saved model")
    args = ap.parse_args()

    df = load_journeys(DATA_FILE)
    data = build_dataset(df)
    fit, val, test, ids = split_by_journey(data)
    print(f"train journeys {len(ids[0])} | val {len(ids[1])} | test {len(ids[2])} "
          f"({len(test)} unseen rows)")
    meta = {"model_type": "XGBRegressor", "trained_by": "preserved original model"}

    if args.retrain:
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=4, min_child_weight=5,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                         early_stopping_rounds=50, random_state=SEED)
        m.fit(model_matrix(fit), fit["target_change"],
              eval_set=[(model_matrix(val), val["target_change"])], verbose=False)
        m.save_model(str(MODEL_FILE)); joblib.dump(FEATURE_COLUMNS, FEATURE_FILE)
        meta = {"model_type": "XGBRegressor", "trained_by": "src.training --retrain",
                "best_iteration": int(m.best_iteration)}
        print("retrained; best iteration", m.best_iteration)

    svc = ModelService()
    res = evaluate(svc, data, test, ids, meta)
    res["hyperparameters"] = {k: v for k, v in svc.model.get_params().items()
                              if k in ("n_estimators", "learning_rate", "max_depth", "min_child_weight",
                                       "subsample", "colsample_bytree", "reg_lambda")}
    METRICS_FILE.write_text(json.dumps(res, indent=2))
    svc.feature_importance().sort_values(ascending=False).rename("importance").to_csv(IMPORTANCE_FILE)
    for k in ("xgboost", "persistence_baseline", "xgboost_change_only"):
        print(f"{k:<22}", {a: round(b, 3) for a, b in res[k].items()})


if __name__ == "__main__":
    main()
