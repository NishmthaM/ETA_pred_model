"""Per-journey state. One JourneyState == one train run on one date; nothing is shared
between instances, so delays/history can never leak from one journey into another."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import OBSERVED_COLUMNS, TIMETABLE_COLUMNS
from .data_loader import clock_to_minutes


class StateError(ValueError):
    pass


@dataclass
class JourneyState:
    journey_id: str
    journey_date: pd.Timestamp
    train_id: int | str
    timetable: pd.DataFrame                       # all stations, TIMETABLE columns only
    observations: list[dict] = field(default_factory=list)   # one dict per reached station

    def __post_init__(self) -> None:
        if self.timetable.empty:
            raise StateError("Timetable is empty.")
        self.timetable = self.timetable[TIMETABLE_COLUMNS].reset_index(drop=True).copy()

    # ---- position ------------------------------------------------------------------
    @property
    def n_stations(self) -> int:
        return len(self.timetable)

    @property
    def current_idx(self) -> int:
        """Index of the last station with an observation (-1 = journey not started)."""
        return len(self.observations) - 1

    @property
    def has_next(self) -> bool:
        return 0 <= self.current_idx < self.n_stations - 1

    def next_station_row(self) -> pd.Series:
        if not self.has_next:
            raise StateError("There is no next station.")
        return self.timetable.iloc[self.current_idx + 1]

    # ---- updates -------------------------------------------------------------------
    def observe(self, station_code: str, obs: dict) -> None:
        """Record actual data for the next station in sequence."""
        idx = self.current_idx + 1
        if idx >= self.n_stations:
            raise StateError("Journey already finished.")
        expected = self.timetable.loc[idx, "station_code"]
        if station_code != expected:
            raise StateError(f"Out-of-sequence observation: expected {expected}, got {station_code}.")
        missing = [c for c in OBSERVED_COLUMNS if c not in obs or pd.isna(obs[c])]
        if missing:
            raise StateError(f"Observation for {station_code} is missing values: {missing}")
        self.observations.append({c: obs[c] for c in OBSERVED_COLUMNS})

    # ---- view used by the predictor ---------------------------------------------------
    def history_frame(self) -> pd.DataFrame:
        """Rows for stations 0..current only (timetable + observed values)."""
        if not self.observations:
            raise StateError("No observations yet.")
        k = len(self.observations)
        frame = self.timetable.iloc[:k].reset_index(drop=True).copy()
        frame = pd.concat([frame, pd.DataFrame(self.observations)], axis=1)
        frame["journey_id"] = self.journey_id
        frame["journey_date"] = self.journey_date
        frame["train_id"] = self.train_id
        frame["station_idx"] = range(k)
        frame["sched_min"] = frame["Scheduled_arrival"].map(clock_to_minutes)
        frame["preceding_delay_min"] = frame["preceding_train_delay_min"].map(clock_to_minutes)
        return frame

