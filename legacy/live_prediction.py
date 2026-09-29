
import numpy as np
import pandas as pd
import joblib

from xgboost import XGBRegressor


# ==========================================
# 1. LOAD THE TRAINED MODEL
# ==========================================

model = XGBRegressor()

model.load_model(
    "railway_eta_xgb.json"
)

features = joblib.load(
    "railway_xgb_features.pkl"
)


# ==========================================
# 2. PREDICT THE NEXT STATION DELAY
# ==========================================

def predict_next_station(
    current_delay,
    delay_lag1,
    delay_lag2,
    mean_delay_so_far,
    stations_done,
    speed,
    distance_from_origin,
    distance_to_next,
    scheduled_travel,
    dwell_time,
    preceding_delay,
    congestion,
    congestion_lag1,
    rain,
    scheduled_hour,
    month,
    weekday
):

    # Create features using the same
    # definitions used during training.

    delay_change = (
        current_delay - delay_lag1
    )

    delay_trend2 = (
        current_delay - delay_lag2
    )

    scheduled_minutes = (
        scheduled_hour * 60
    )

    sched_sin = np.sin(
        2 * np.pi * scheduled_minutes / 1440
    )

    sched_cos = np.cos(
        2 * np.pi * scheduled_minutes / 1440
    )

    row = {
        "current_delay_min": current_delay,
        "delay_lag1": delay_lag1,
        "delay_lag2": delay_lag2,
        "delay_change": delay_change,
        "delay_trend2": delay_trend2,
        "mean_delay_so_far": mean_delay_so_far,
        "stations_done": stations_done,
        "speed_kmph": speed,
        "distance_from_origin_km": distance_from_origin,
        "distance_to_next_km": distance_to_next,
        "scheduled_travel_min": scheduled_travel,
        "dwell_time_min": dwell_time,
        "preceding_delay_min": preceding_delay,
        "congestion": congestion,
        "cong_lag1": congestion_lag1,
        "rain_mm": rain,
        "sched_sin": sched_sin,
        "sched_cos": sched_cos,
        "month": month,
        "weekday": weekday
    }

    # Match the exact feature order
    # used by the trained model.

    input_data = pd.DataFrame(
        [row]
    )[features]

    # Model predicts change in delay
    predicted_change = model.predict(
        input_data
    )[0]

    # Convert change into next-station delay
    predicted_delay = (
        current_delay + predicted_change
    )

    # Delay cannot be negative
    predicted_delay = max(
        0,
        predicted_delay
    )

    return round(
        float(predicted_delay),
        2
    )


# ==========================================
# 3. EXAMPLE
# ==========================================

# Example inputs only.
# Replace these values with actual
# live train information.

predicted_delay = predict_next_station(
    current_delay=10,
    delay_lag1=8,
    delay_lag2=5,
    mean_delay_so_far=7,
    stations_done=5,
    speed=45,
    distance_from_origin=120,
    distance_to_next=25,
    scheduled_travel=30,
    dwell_time=2,
    preceding_delay=8,
    congestion=2,
    congestion_lag1=1,
    rain=10,
    scheduled_hour=12,
    month=9,
    weekday=1
)

print(
    "Predicted next-station delay:",
    predicted_delay,
    "minutes"
)