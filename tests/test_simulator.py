import numpy as np
import pytest

from src.simulator import JourneySimulator
from src.prediction_service import predict_next
import src.simulator as simmod


def test_starts_with_only_origin_revealed(journey, model):
    sim = JourneySimulator(journey, model)
    t = sim.station_table()
    assert sim.state.current_idx == 0 and len(sim.predictions) == 1
    assert t.actual_delay.notna().sum() == 1 and t.actual_arrival.notna().sum() == 1
    assert t.pred_delay.notna().sum() == 1                   # only station 1 predicted so far


def test_each_step_history_contains_only_reached_stations(journey, model, monkeypatch):
    seen = []
    def spy(state, m):
        h = state.history_frame()
        seen.append((state.current_idx, len(h), h.station_idx.max()))
        return predict_next(state, m)
    monkeypatch.setattr(simmod, "predict_next", spy)
    sim = JourneySimulator(journey, model)
    while sim.advance():
        pass
    assert seen and all(k == n - 1 == mx for k, n, mx in seen)
    assert len(seen) == 20                                   # stations 0..19 each predict the next


def test_future_actuals_cannot_influence_past_predictions(journey, model):
    """Corrupt every FUTURE actual value; predictions made up to station k must not change."""
    k = 8
    a = JourneySimulator(journey, model)
    for _ in range(k):
        a.advance()
    j2 = journey.copy()
    cols = ["current_delay_min", "congestion", "dwell_time_min", "speed_kmph", "rain_mm", "target_delay_min"]
    j2.loc[j2.station_idx > k, cols] = 999
    b = JourneySimulator(j2, model)
    for _ in range(k):
        b.advance()
    assert [p.predicted_delay_min for p in a.predictions] == [p.predicted_delay_min for p in b.predictions]


def test_target_column_never_used_as_feature(journey, model):
    sim = JourneySimulator(journey, model)
    assert "target_delay_min" not in sim.predictions[0].features_used
    assert "target_delay_min" not in sim.state.history_frame().columns


def test_actual_revealed_only_on_arrival(journey, model):
    sim = JourneySimulator(journey, model)
    sim.advance(); sim.advance()
    t = sim.station_table()
    assert list(t.status[:4]) == ["reached", "reached", "current", "upcoming"]
    assert t.actual_delay[:3].notna().all() and t.actual_delay[3:].isna().all()
    assert t.actual_delay[2] == journey.current_delay_min[2]


def test_reset_and_isolation(df, model, journey):
    from src.data_loader import get_journey
    a = JourneySimulator(journey, model)
    b = JourneySimulator(get_journey(df, "J0010"), model)
    b0 = [p.predicted_delay_min for p in b.predictions]
    for _ in range(6):
        a.advance()
    assert [p.predicted_delay_min for p in b.predictions] == b0 and b.state.current_idx == 0
    first = a.predictions[0].predicted_delay_min
    a.reset()
    assert a.state.current_idx == 0 and len(a.predictions) == 1 and a.predictions[0].predicted_delay_min == first


def test_full_run_and_finish(journey, model):
    sim = JourneySimulator(journey, model)
    steps = 0
    while sim.advance():
        steps += 1
    assert steps == 20 and sim.finished and sim.latest_prediction is None
    assert sim.advance() is False
    t = sim.station_table()
    assert t.actual_delay.notna().all() and t.error.notna().sum() == 20


def test_rejects_multi_journey_input(df, model):
    with pytest.raises(ValueError):
        JourneySimulator(df[df.journey_id.isin(["J0001", "J0002"])], model)
