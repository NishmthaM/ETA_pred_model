import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import get_journey, load_journeys  # noqa: E402
from src.model_service import ModelService  # noqa: E402


@pytest.fixture(scope="session")
def df():
    return load_journeys()


@pytest.fixture(scope="session")
def model():
    return ModelService()


@pytest.fixture()
def journey(df):
    return get_journey(df, "J0450")
