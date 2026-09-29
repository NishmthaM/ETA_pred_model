"""Model loading + prediction. The model predicts the CHANGE in delay between the current
and the next station (target = target_delay_min - current_delay_min)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .config import FEATURE_FILE, MODEL_FILE
from .feature_engineering import FEATURE_COLUMNS, model_matrix


class ModelError(RuntimeError):
    pass


class ModelService:
    def __init__(self, model_path: Path = MODEL_FILE, feature_path: Path = FEATURE_FILE):
        for p in (model_path, feature_path):
            if not Path(p).exists():
                raise ModelError(f"Missing model artifact: {p}. Run `python -m src.training`.")
        self.model = XGBRegressor()
        self.model.load_model(str(model_path))
        self.features: list[str] = list(joblib.load(feature_path))
        if self.features != FEATURE_COLUMNS:
            raise ModelError("Saved feature list differs from feature_engineering.FEATURE_COLUMNS "
                             "(names or order). Retrain the model.")
        if self.model.n_features_in_ != len(self.features):
            raise ModelError("Model input width does not match the feature list.")

    def predict_change(self, featured: pd.DataFrame) -> np.ndarray:
        """Predicted delay change (minutes) for each row of an already-featured frame."""
        return self.model.predict(model_matrix(featured, self.features))

    @staticmethod
    def to_next_delay(current_delay: np.ndarray, change: np.ndarray) -> np.ndarray:
        """Next-station delay = current delay + predicted change, floored at 0 (early running
        is not modelled: the training data has no negative delays)."""
        return np.clip(np.asarray(current_delay, dtype=float) + change, 0, None)

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.features).sort_values()
