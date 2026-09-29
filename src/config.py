"""Central configuration. All paths are anchored to the project root, so the
app works regardless of the terminal's current working directory."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "synthetic" / "railway_eta_synthetic_500_journeys.xlsx"
MODEL_FILE = PROJECT_ROOT / "models" / "railway_eta_xgb.json"
FEATURE_FILE = PROJECT_ROOT / "models" / "railway_xgb_features.pkl"
METRICS_FILE = PROJECT_ROOT / "models" / "metrics.json"
IMPORTANCE_FILE = PROJECT_ROOT / "models" / "feature_importance.csv"
CSS_FILE = PROJECT_ROOT / "assets" / "styles" / "theme.css"

SEED = 42
TEST_FRACTION = 0.20          # last 20 % of journeys (chronological) = unseen test set
VALID_FRACTION = 0.15         # last 15 % of TRAIN journeys = early-stopping set

# Columns that come from the published timetable (known before the train runs).
TIMETABLE_COLUMNS = [
    "station_full_name", "station_code", "Scheduled_arrival",
    "distance_from_origin_km", "distance_to_next_km", "scheduled_travel_min",
]
# Columns that are only known once the train has actually reached / left a station.
OBSERVED_COLUMNS = [
    "current_delay_min", "speed_kmph", "dwell_time_min",
    "preceding_train_delay_min", "congestion", "rain_mm",
]

DATA_SOURCE_LABEL = "SYNTHETIC (simulated) - not real Indian Railways data"
