from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _app():
    return AppTest.from_file(APP, default_timeout=90).run()


def test_app_starts_without_exception():
    at = _app()
    assert not at.exception and len(at.tabs) == 6


def test_next_and_reset_buttons():
    at = _app()
    next(b for b in at.button if b.label.startswith("⏭")).click(); at.run()
    assert not at.exception and at.session_state["sim"].state.current_idx == 1
    next(b for b in at.button if b.label.startswith("↺")).click(); at.run()
    assert at.session_state["sim"].state.current_idx == 0


def test_live_mode_reports_unavailable():
    at = _app()
    at.sidebar.radio[0].set_value("Real data / live API").run()
    assert not at.exception and any("No authorised live data source" in w.value for w in at.warning)
