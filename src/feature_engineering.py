"""The ONE feature pipeline, shared by training, evaluation and live inference.

Every operation is strictly *causal* (uses only the current and earlier stations of the
same journey), so computing features on a full journey and on a journey revealed only up
to station k gives identical values for station k. tests/test_features.py enforces this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS: list[str] = [
    "current_delay_min", "delay_lag1", "delay_lag2", "delay_change", "delay_trend2",
    "mean_delay_so_far", "stations_done", "speed_kmph", "distance_from_origin_km",
    "distance_to_next_km", "scheduled_travel_min", "dwell_time_min", "preceding_delay_min",
    "congestion", "cong_lag1", "rain_mm", "sched_sin", "sched_cos", "month", "weekday",
]

# Raw columns that add_features() needs (all must be present, non-null, at the current station).
BASE_COLUMNS = [
    "journey_id", "journey_date", "station_idx", "sched_min", "current_delay_min",
    "congestion", "speed_kmph", "distance_from_origin_km", "distance_to_next_km",
    "scheduled_travel_min", "dwell_time_min", "preceding_delay_min", "rain_mm",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all model features. Input must be ordered by station within each journey.

    Missing history is left as NaN (e.g. delay_lag1 at the first station). XGBoost handles
    NaN natively and was trained with NaN there, so no fake default is injected.
    """
    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"add_features: missing input columns {missing}")
    out = df.sort_values(["journey_id", "station_idx"], kind="stable").copy()
    g = out.groupby("journey_id", sort=False)

    out["delay_lag1"] = g["current_delay_min"].shift(1)
    out["delay_lag2"] = g["current_delay_min"].shift(2)
    out["delay_change"] = out["current_delay_min"] - out["delay_lag1"]
    out["delay_trend2"] = out["current_delay_min"] - out["delay_lag2"]
    out["cong_lag1"] = g["congestion"].shift(1)
    # expanding mean INCLUDING the current station (same as the original training script)
    out["mean_delay_so_far"] = g["current_delay_min"].transform(lambda s: s.expanding().mean())
    out["stations_done"] = g.cumcount()

    angle = 2 * np.pi * out["sched_min"] / 1440.0
    out["sched_sin"] = np.sin(angle)
    out["sched_cos"] = np.cos(angle)
    dates = pd.to_datetime(out["journey_date"])
    out["month"] = dates.dt.month
    out["weekday"] = dates.dt.weekday
    return out


def model_matrix(featured: pd.DataFrame, feature_names: list[str] | None = None) -> pd.DataFrame:
    """Select model inputs in the exact trained order, as float64."""
    names = feature_names or FEATURE_COLUMNS
    missing = [c for c in names if c not in featured.columns]
    if missing:
        raise KeyError(f"Missing model features: {missing}")
    return featured[names].astype("float64")
