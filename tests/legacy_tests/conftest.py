"""
The v0.1.9 tests modify the global decayangle configuration (e.g. ``sorting = "off"``),
partly at import time. Isolate those changes so that they cannot leak into the
current test suite, which is collected and run in the same process.
"""
import pytest
from decayangle.config import config as decayangle_config

_STATE = type(decayangle_config)._cfg__state
_SNAPSHOT_BEFORE_COLLECTION = dict(_STATE)


def _restore(snapshot: dict) -> None:
    _STATE.clear()
    _STATE.update(snapshot)


def pytest_collection_finish(session):
    # undo config changes made by importing the legacy test modules
    _restore(_SNAPSHOT_BEFORE_COLLECTION)


@pytest.fixture(autouse=True)
def _isolate_decayangle_config():
    snapshot = dict(_STATE)
    yield
    _restore(snapshot)
