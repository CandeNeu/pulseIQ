"""Headless smoke tests for the Streamlit app via st.testing.v1.AppTest.

These do not exercise a real camera (the WebRTC component needs a browser);
they check that the script renders each acquisition state without raising,
that the per-session pulse objects are wired up, and that a captured reading
is reported and unlocks the assessment step.
"""

import pathlib

from streamlit.testing.v1 import AppTest

from ppg import BAND_HIGH_HZ, BAND_LOW_HZ, PulseSampler, PulseTracker

APP = str(pathlib.Path(__file__).resolve().parents[1] / "frontend" / "app.py")

CAPTURED = {
    "bpm": 72.4,
    "quality": 84.0,
    "source": "Live",
    "status": "Stable",
    "duration_s": 12.0,
    "fps": 29.6,
    "channel": "Red",
    "spectral_bpm": 72.3,
    "ibi_bpm": 72.5,
    "n_beats": 14,
    "beat_consistency": 0.93,
    "captured_at": "14:02:11 UTC",
}


def run_app(seed=None):
    at = AppTest.from_file(APP, default_timeout=30)
    for key, value in (seed or {}).items():
        at.session_state[key] = value
    return at.run()


def click(at, label):
    return button(at, label).click().run()


def captions(at):
    return [str(c.value) for c in at.caption]


def button(at, label):
    return next(b for b in at.button if b.label == label)


def test_patient_step_renders_without_errors():
    at = run_app()
    assert not at.exception
    assert at.session_state["step"] == 1


def test_begin_acquisition_creates_per_session_pulse_objects():
    at = click(run_app(), "Begin acquisition")
    assert not at.exception
    assert at.session_state["step"] == 2
    assert isinstance(at.session_state["pulse_sampler"], PulseSampler)
    assert isinstance(at.session_state["pulse_tracker"], PulseTracker)


def test_acquisition_step_starts_idle_with_assessment_disabled():
    at = click(run_app(), "Begin acquisition")
    assert any("no signal" in c.lower() for c in captions(at))
    assert button(at, "Run assessment").disabled
    assert not at.success


def test_live_camera_mode_renders_without_errors():
    # The WebRTC component cannot start without a browser session; the app
    # must degrade to an error panel plus the idle monitor, not a stack trace.
    at = click(run_app(), "Begin acquisition")
    mode = next(s for s in at.segmented_control if s.label == "Acquisition mode")
    mode.set_value("Live camera").run()
    assert not at.exception
    assert any("camera stopped" in c.lower() for c in captions(at))


def test_captured_reading_is_reported_and_unlocks_assessment():
    at = run_app(
        seed={
            "step": 2,
            "measured_bpm": CAPTURED["bpm"],
            "signal_pct": CAPTURED["quality"],
            "source": "Live",
            "analysis": CAPTURED,
        }
    )
    assert not at.exception
    assert not button(at, "Run assessment").disabled
    assert at.success, "expected a 'reading accepted' banner"
    summary = " ".join(str(t.value) for t in at.table)
    for needle in ("Spectral rate", "Beat-interval rate", "72.3", "14 beats", "Red"):
        assert needle in summary


def test_run_assessment_from_captured_reading_reaches_step_three():
    at = run_app(
        seed={
            "step": 2,
            "measured_bpm": CAPTURED["bpm"],
            "signal_pct": CAPTURED["quality"],
            "source": "Live",
            "analysis": CAPTURED,
        }
    )
    click(at, "Run assessment")
    assert at.session_state["step"] == 3


def test_sidebar_passband_matches_pipeline_constants():
    at = run_app()
    text = " ".join(str(t.value) for t in at.sidebar.table)
    assert f"{BAND_LOW_HZ}–{BAND_HIGH_HZ} Hz" in text
