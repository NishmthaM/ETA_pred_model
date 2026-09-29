"""Loading and validating the journey dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_FILE

REQUIRED_COLUMNS = [
    "journey_id", "journey_date", "train_id", "station_full_name", "station_code",
    "speed_kmph", "Scheduled_arrival", "current_delay_min", "distance_from_origin_km",
    "distance_to_next_km", "scheduled_travel_min", "dwell_time_min",
    "preceding_train_delay_min", "congestion", "rain_mm", "target_delay_min",
]


class DataError(ValueError):
    """Raised when the dataset is missing, malformed or inconsistent."""


def clock_to_minutes(value) -> float:
    """'HH:MM[:SS]' / datetime.time / Timestamp -> minutes after midnight (NaN if missing).

    The dataset stores `preceding_train_delay_min` as a clock string such as 00:02:00
    (= 2 minutes); the same conversion is therefore used for it.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return np.nan
    if hasattr(value, "hour") and hasattr(value, "minute"):
        return float(value.hour * 60 + value.minute)
    parts = str(value).split(":")
    if len(parts) < 2:
        raise DataError(f"Cannot parse time value {value!r}")
    return float(int(parts[0]) * 60 + int(parts[1]))


def minutes_to_clock(minutes: float) -> str:
    """Minutes after midnight -> 'HH:MM' (wraps past midnight, e.g. 1450 -> '00:10')."""
    m = int(round(minutes)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Dataset is missing required columns: {missing}")


def prepare_journeys(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy: numeric minute columns, explicit station order per journey.

    Station order is taken from the file's row order *within each journey* (the file is
    written station by station); it is never inferred from sorting by time or code.
    """
    validate_columns(raw)
    if raw.empty:
        raise DataError("Dataset is empty.")
    df = raw.copy()
    df["journey_date"] = pd.to_datetime(df["journey_date"])
    df["sched_min"] = df["Scheduled_arrival"].map(clock_to_minutes)
    df["preceding_delay_min"] = df["preceding_train_delay_min"].map(clock_to_minutes)
    if df["sched_min"].isna().any():
        raise DataError("Scheduled_arrival contains missing/invalid values.")
    df["station_idx"] = df.groupby("journey_id", sort=False).cumcount()
    df["journey_pos"] = pd.factorize(df["journey_id"])[0]      # file order of journeys
    df = df.sort_values(["journey_date", "journey_pos", "station_idx"]).reset_index(drop=True)
    df["actual_arrival_min"] = df["sched_min"] + df["current_delay_min"]
    return df


def load_journeys(path: str | Path = DATA_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataError(f"Dataset not found: {path}. Run `python -m scripts.generate_synthetic`.")
    raw = pd.read_excel(path)
    return prepare_journeys(raw)


def get_journey(df: pd.DataFrame, journey_id: str) -> pd.DataFrame:
    j = df[df["journey_id"] == journey_id].sort_values("station_idx")
    if j.empty:
        raise DataError(f"Unknown journey_id {journey_id!r}")
    return j.reset_index(drop=True)
