import numpy as np
import pytest

from src.feature_engineering import add_features
from src.journey_state import JourneyState, StateError
from src.config import OBSERVED_COLUMNS, TIMETABLE_COLUMNS
from src.prediction_service import predict_next
from src.simulator import JourneySimulator


def _state_upto(journey, k):
    st = JourneyState("J", journey.journey_date[0], 10108, journey[TIMETABLE_COLUMNS])
    for i in range(k + 1):
        r = journey.iloc[i]
        st.observe(r.station_code, {c: r[c] for c in OBSERVED_COLUMNS})
    return st


def test_prediction_matches_batch_pipeline(journey, model):
    """Live path == training-table path for the same station."""
    full = add_features(journey)
    for k in (0, 3, 10, 19):
        p = predict_next(_state_upto(journey, k), model)
        batch = model.predict_change(full.iloc[[k]])[0]
        assert p.predicted_change_min == pytest.approx(float(batch), abs=1e-5)


def test_prediction_output_and_eta(journey, model):
    p = predict_next(_state_upto(journey, 5), model)
    assert isinstance(p.predicted_delay_min, float) and p.predicted_delay_min >= 0
    assert p.next_station_code == journey.station_code[6]
    assert p.scheduled_arrival_next_min == 60 * 17 + 26          # BKJ scheduled 17:26
    assert p.predicted_arrival_next_min == pytest.approx(p.scheduled_arrival_next_min + p.predicted_delay_min)
    assert list(p.features_used)[:2] == ["current_delay_min", "delay_lag1"]


def test_deterministic(journey, model):
    s = _state_upto(journey, 7)
    assert predict_next(s, model) == predict_next(s, model)


def test_final_station_and_unstarted_raise(journey, model):
    with pytest.raises(StateError):
        predict_next(_state_upto(journey, 20), model)
    st = JourneyState("J", journey.journey_date[0], 1, journey[TIMETABLE_COLUMNS])
    with pytest.raises(StateError):
        predict_next(st, model)


def test_out_of_sequence_and_missing_values_rejected(journey):
    st = JourneyState("J", journey.journey_date[0], 1, journey[TIMETABLE_COLUMNS])
    r = journey.iloc[0]
    with pytest.raises(StateError):
        st.observe("XXX", {c: r[c] for c in OBSERVED_COLUMNS})
    bad = {c: r[c] for c in OBSERVED_COLUMNS}; bad["congestion"] = np.nan
    with pytest.raises(StateError):
        st.observe(r.station_code, bad)
    assert st.current_idx == -1                              # failed observation stored nothing


def test_predicted_delay_never_negative(df, model):
    j = df[df.journey_id == "J0300"].reset_index(drop=True)
    sim = JourneySimulator(j, model)
    while sim.advance():
        pass
    assert all(p.predicted_delay_min >= 0 for p in sim.predictions)
