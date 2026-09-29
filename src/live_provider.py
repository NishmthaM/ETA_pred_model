"""MODE 2 - REAL DATA. A provider interface only; NO real API is implemented or invented.

To go live, subclass LiveDataProvider and return an `Observation` from a source you are
authorised to use (e.g. an official Indian Railways/NTES data agreement, authorised GPS/AVL
feeds). Then feed each observation to JourneyState.observe() and call predict_next().
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    station_code: str
    current_delay_min: float
    speed_kmph: float
    dwell_time_min: float
    preceding_train_delay_min: str        # 'HH:MM' clock-style, as in the dataset
    congestion: int                       # 0-3 scale used in training
    rain_mm: float


class LiveDataProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def latest_observation(self, train_id: str, journey_date: str) -> Observation:
        """Return the newest verified observation for a running train."""


class UnavailableProvider(LiveDataProvider):
    name = "none configured"

    def is_available(self) -> bool:
        return False

    def latest_observation(self, train_id: str, journey_date: str) -> Observation:
        raise NotImplementedError(self.instructions())

    @staticmethod
    def instructions() -> str:
        return ("No authorised live data source is configured. To enable live mode: "
                "(1) obtain legitimate API/GPS access, (2) implement LiveDataProvider in "
                "src/live_provider.py, (3) map its fields to Observation (all six fields are "
                "needed - the model was trained on them), (4) register it in app.py. "
                "Also required: live congestion and preceding-train delay, which most public "
                "feeds do not provide - see README 'Going live'.")


def get_provider() -> LiveDataProvider:
    return UnavailableProvider()
