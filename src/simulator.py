"""DEMO mode: replay one recorded (synthetic) journey station by station.

The simulator owns the full recorded journey privately (`_truth`) but only hands a station's
observed values to the JourneyState when that station is "reached". Predictions are made from
the state alone, so the model can never see future actual delays.
"""
from __future__ import annotations

import pandas as pd

from .config import OBSERVED_COLUMNS, TIMETABLE_COLUMNS
from .data_loader import minutes_to_clock
from .journey_state import JourneyState
from .model_service import ModelService
from .prediction_service import Prediction, predict_next


class JourneySimulator:
    def __init__(self, journey: pd.DataFrame, model: ModelService):
        if journey["journey_id"].nunique() != 1:
            raise ValueError("Simulator needs rows of exactly ONE journey.")
        j = journey.sort_values("station_idx").reset_index(drop=True)
        self._truth = j                                   # private: full recorded journey
        self._model = model
        first = j.iloc[0]
        self.state = JourneyState(first["journey_id"], first["journey_date"], first["train_id"],
                                  j[TIMETABLE_COLUMNS])
        self.predictions: list[Prediction] = []           # predictions[k] = made at station k
        self.reset()

    # ---- control ---------------------------------------------------------------------
    def reset(self) -> None:
        """Back to the start: fresh state, only the origin station is revealed."""
        first = self._truth.iloc[0]
        self.state = JourneyState(first["journey_id"], first["journey_date"], first["train_id"],
                                  self._truth[TIMETABLE_COLUMNS])
        self.predictions = []
        self._reveal(0)
        self._predict()

    @property
    def finished(self) -> bool:
        return not self.state.has_next

    def advance(self) -> bool:
        """Train reaches the next station. Returns False if the journey is already over."""
        if self.finished:
            return False
        self._reveal(self.state.current_idx + 1)
        if self.state.has_next:
            self._predict()
        return True

    # ---- internals ---------------------------------------------------------------------
    def _reveal(self, idx: int) -> None:
        row = self._truth.iloc[idx]
        self.state.observe(row["station_code"], {c: row[c] for c in OBSERVED_COLUMNS})

    def _predict(self) -> None:
        self.predictions.append(predict_next(self.state, self._model))

    # ---- read-only views for the UI -------------------------------------------------
    @property
    def latest_prediction(self) -> Prediction | None:
        return self.predictions[-1] if self.predictions and self.state.has_next else None

    def station_table(self) -> pd.DataFrame:
        """One row per station. Actual columns are None until that station is reached;
        predicted columns exist only for stations after the origin that have been predicted."""
        tt = self.state.timetable
        k = self.state.current_idx
        rows = []
        pred_by_station = {p.next_station_idx: p for p in self.predictions}
        for i in range(len(tt)):
            sched = float(_sched(tt.iloc[i]))
            reached = i <= k
            obs = self.state.observations[i] if reached else None
            p = pred_by_station.get(i)
            rows.append({
                "idx": i, "station": tt.loc[i, "station_full_name"], "code": tt.loc[i, "station_code"],
                "distance_km": tt.loc[i, "distance_from_origin_km"],
                "scheduled": minutes_to_clock(sched),
                "status": "reached" if i < k else "current" if i == k else "upcoming",
                "actual_delay": float(obs["current_delay_min"]) if reached else None,
                "actual_arrival": minutes_to_clock(sched + obs["current_delay_min"]) if reached else None,
                "pred_delay": p.predicted_delay_min if p else None,
                "pred_arrival": p.predicted_arrival_next if p else None,
            })
        t = pd.DataFrame(rows)
        t["error"] = t["pred_delay"] - t["actual_delay"]
        return t


def _sched(row) -> float:
    from .data_loader import clock_to_minutes
    return clock_to_minutes(row["Scheduled_arrival"])
