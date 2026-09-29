"""Next-station prediction from a JourneyState (the only place inference is done)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .data_loader import minutes_to_clock
from .feature_engineering import add_features
from .journey_state import JourneyState, StateError
from .model_service import ModelService


@dataclass(frozen=True)
class Prediction:
    journey_id: str
    current_station: str
    current_station_code: str
    next_station: str
    next_station_code: str
    next_station_idx: int
    current_delay_min: float
    predicted_change_min: float
    predicted_delay_min: float
    scheduled_arrival_next_min: float
    predicted_arrival_next_min: float
    remaining_distance_km: float
    features_used: dict

    @property
    def scheduled_arrival_next(self) -> str:
        return minutes_to_clock(self.scheduled_arrival_next_min)

    @property
    def predicted_arrival_next(self) -> str:
        return minutes_to_clock(self.predicted_arrival_next_min)

    def as_dict(self) -> dict:
        return asdict(self)


def predict_next(state: JourneyState, model: ModelService) -> Prediction:
    """Predict the delay/ETA at the next station using only state.history_frame()
    (stations 0..current). The next station's *timetable* row is used only for its
    scheduled arrival, which is public information."""
    if not state.has_next:
        raise StateError("Cannot predict: train is at its final station (or has not started).")
    hist = state.history_frame()
    featured = add_features(hist)
    row = featured.iloc[[-1]]                        # current station
    change = float(model.predict_change(row)[0])
    cur_delay = float(row["current_delay_min"].iloc[0])
    pred_delay = float(model.to_next_delay(cur_delay, change))

    tt = state.timetable
    cur, nxt = tt.iloc[state.current_idx], state.next_station_row()
    from .data_loader import clock_to_minutes
    sched_next = clock_to_minutes(nxt["Scheduled_arrival"])
    remaining = float(tt["distance_from_origin_km"].iloc[-1] - cur["distance_from_origin_km"])
    return Prediction(
        journey_id=state.journey_id,
        current_station=cur["station_full_name"], current_station_code=cur["station_code"],
        next_station=nxt["station_full_name"], next_station_code=nxt["station_code"],
        next_station_idx=state.current_idx + 1,
        current_delay_min=cur_delay, predicted_change_min=change, predicted_delay_min=pred_delay,
        scheduled_arrival_next_min=sched_next,
        predicted_arrival_next_min=sched_next + pred_delay,
        remaining_distance_km=remaining,
        features_used={k: (None if v != v else float(v)) for k, v in
                       row[model.features].iloc[0].items()},
    )
