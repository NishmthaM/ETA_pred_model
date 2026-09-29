import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_FILE
from src.data_loader import DataError, clock_to_minutes, minutes_to_clock, prepare_journeys
from src.feature_engineering import FEATURE_COLUMNS, add_features, model_matrix


def test_feature_order_matches_saved_model():
    assert list(joblib.load(FEATURE_FILE)) == FEATURE_COLUMNS


def test_time_conversions():
    assert clock_to_minutes("15:30:00") == 930
    assert clock_to_minutes("00:02:00") == 2
    assert np.isnan(clock_to_minutes(None)) and np.isnan(clock_to_minutes(np.nan))
    assert minutes_to_clock(930) == "15:30"
    assert minutes_to_clock(1450) == "00:10"        # wraps past midnight
    with pytest.raises(DataError):
        clock_to_minutes("garbage")


def test_matches_original_training_script_features(df):
    """Re-implements the ORIGINAL train_xgboost.py feature code and compares."""
    d = df.copy()
    g = d.groupby("journey_id", sort=False)
    ref = pd.DataFrame({
        "delay_lag1": g["current_delay_min"].shift(1), "delay_lag2": g["current_delay_min"].shift(2),
        "cong_lag1": g["congestion"].shift(1),
        "mean_delay_so_far": g["current_delay_min"].transform(lambda s: s.expanding().mean()),
        "stations_done": g.cumcount(),
    })
    ref["sched_sin"] = np.sin(2 * np.pi * d["sched_min"] / 1440)
    ref["month"] = d["journey_date"].dt.month
    ref["weekday"] = d["journey_date"].dt.weekday
    f = add_features(df).reset_index(drop=True)
    for c in ref.columns:
        np.testing.assert_allclose(f[c].astype(float), ref[c].astype(float), equal_nan=True, err_msg=c)


def test_first_station_history_is_nan_not_faked(df):
    f = add_features(df)
    first = f[f.station_idx == 0]
    assert first["delay_lag1"].isna().all() and first["delay_lag2"].isna().all()


def test_features_are_causal(df):
    """Features at station k must not change when later stations are removed."""
    j = df[df.journey_id == "J0100"].sort_values("station_idx")
    full = add_features(j)
    for k in (0, 1, 2, 9, 19):
        part = add_features(j.iloc[: k + 1])
        pd.testing.assert_series_equal(
            model_matrix(part).iloc[-1], model_matrix(full).iloc[k], check_names=False)


def test_journeys_do_not_mix(df):
    a, b = df[df.journey_id == "J0001"], df[df.journey_id == "J0002"]
    both = add_features(pd.concat([a, b]))
    alone = add_features(b)
    x = model_matrix(both[both.journey_id == "J0002"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(x, model_matrix(alone).reset_index(drop=True))


def test_missing_columns_raise(df):
    with pytest.raises(KeyError):
        add_features(df.drop(columns=["congestion"]))
    with pytest.raises(DataError):
        prepare_journeys(df.drop(columns=["journey_id"]))
    with pytest.raises(DataError):
        prepare_journeys(df.iloc[0:0])
