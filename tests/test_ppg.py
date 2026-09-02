"""Tests for the PPG pipeline in frontend/ppg.py, driven by synthetic signals.

The synthetic generator models what a fingertip on a camera actually
produces: a pulse with a second harmonic (dicrotic notch), slow baseline
wander, an auto-exposure settling transient, white noise, dropped frames and
timing jitter. Every test states the ground-truth heart rate it expects back.
"""

import asyncio
import fractions

import av
import numpy as np
import pytest
from scipy.signal import butter, filtfilt

from ppg import (
    BAND_HIGH_HZ,
    BAND_LOW_HZ,
    FrameSample,
    PulseSampler,
    PulseTracker,
    coverage_score,
    estimate_pulse,
    sample_frame,
    samples_from_video,
)

# ============================== Synthetic PPG ==============================


def synth_ppg(
    hr_bpm,
    duration_s=12.0,
    fps=30.0,
    *,
    amp=3.0,
    harmonic=0.35,
    noise=0.0,
    drift=0.0,
    settle=0.0,
    step_at=None,
    step_size=0.0,
    drop_frac=0.0,
    jitter_s=0.0,
    pulse_channel="r",
    saturate=None,
    weak_channel_noise=0.3,
    seed=0,
):
    """Return a list of FrameSample for a synthetic fingertip recording.

    The pulse lives in `pulse_channel`; the other channels carry a weak copy
    of it (0.2x) plus independent noise, as real skin does.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fps))
    t = np.arange(n) / fps
    if jitter_s:
        t = t + rng.uniform(-jitter_s, jitter_s, n)
        t = np.sort(t)

    f = hr_bpm / 60.0
    phase = 2 * np.pi * f * t
    pulse = np.cos(phase) + harmonic * np.cos(2 * phase + 0.6)
    baseline = drift * np.sin(2 * np.pi * 0.15 * t) + settle * np.exp(-t / 1.5)
    if step_at is not None:
        baseline = baseline + step_size * (t >= step_at)

    # Sensor noise is at least ~0.3 levels in every channel, so the channels
    # carrying only a weak copy of the pulse are relatively noisier, as in a
    # real camera.
    base = {"r": 120.0, "g": 60.0, "b": 40.0}
    channels = {}
    for name in ("r", "g", "b"):
        gain = amp if name == pulse_channel else 0.2 * amp
        sigma = noise if name == pulse_channel else max(noise, weak_channel_noise)
        channels[name] = (
            base[name] + gain * pulse + baseline + rng.normal(0.0, sigma, n)
        )
    if saturate:
        channels[saturate] = np.minimum(255.0, 254.6 + rng.normal(0, 0.2, n))

    keep = rng.random(n) >= drop_frac if drop_frac else np.ones(n, dtype=bool)
    keep[0] = True
    return [
        FrameSample(
            t=float(t[i]),
            r=float(channels["r"][i]),
            g=float(channels["g"][i]),
            b=float(channels["b"][i]),
            r_std=5.0,
            saturation=1.0 if saturate == "r" else 0.0,
        )
        for i in range(n)
        if keep[i]
    ]


def legacy_bpm(red_means, fps=30.0):
    """The estimator this work replaces (frontend/app.py before this change)."""
    signal = np.array(red_means, dtype=np.float32)
    if len(signal) < fps * 3:
        return None
    signal = signal - signal.mean()
    nyq = fps / 2.0
    b, a = butter(3, [0.7 / nyq, 4.0 / nyq], btype="band")
    filtered = filtfilt(b, a, signal)
    fft = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fps)
    valid = (freqs >= 0.7) & (freqs <= 4.0)
    return round(freqs[valid][np.argmax(fft[valid])] * 60.0, 1)


def red_frame(r=150, g=60, b=40, shape=(48, 64)):
    img = np.zeros((*shape, 3), dtype=np.uint8)
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


def scene_frame(shape=(48, 64), seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(*shape, 3), dtype=np.uint8)


def face_frame(shape=(48, 64)):
    """Skin-toned frame with large-scale structure: lit left half, shaded right."""
    img = np.zeros((*shape, 3), dtype=np.uint8)  # BGR
    half = shape[1] // 2
    img[:, :half] = (110, 130, 180)
    img[:, half:] = (50, 60, 90)
    return img


class FakeClock:
    def __init__(self, step):
        self.now = 100.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


# ============================== sample_frame ==============================


def test_sample_frame_averages_centre_roi_in_given_channel_order():
    img = np.zeros((100, 200, 3), dtype=np.uint8)  # BGR
    img[:, :, 2] = 50  # red border
    img[25:75, 50:150, 2] = 200  # central 50% x 50%
    s = sample_frame(img, t=1.5, channel_order="bgr")
    assert s.t == 1.5
    assert s.r == 200.0 and s.g == 0.0 and s.b == 0.0

    s2 = sample_frame(img[:, :, ::-1], t=0.0, channel_order="rgb")
    assert s2.r == 200.0


def test_sample_frame_reports_saturation_and_uniformity():
    s = sample_frame(red_frame(r=255), t=0.0)
    assert s.saturation == 1.0
    assert s.r_std == 0.0
    assert sample_frame(scene_frame(), t=0.0).r_std > 20


# ============================== coverage_score ==============================


def test_coverage_high_for_fingertip_low_for_scene_and_face():
    covered = FrameSample(t=0, r=150, g=60, b=40, r_std=8, saturation=0)
    dark_finger = FrameSample(t=0, r=40, g=15, b=10, r_std=4, saturation=0)
    flash_finger = FrameSample(t=0, r=255, g=150, b=80, r_std=3, saturation=0.9)
    scene = FrameSample(t=0, r=120, g=110, b=100, r_std=60, saturation=0)
    # Lit skin is mildly red-dominant but a face is never spatially uniform.
    face = FrameSample(t=0, r=170, g=120, b=100, r_std=50, saturation=0)
    close_face = FrameSample(t=0, r=180, g=115, b=95, r_std=36, saturation=0)
    assert coverage_score(covered) > 0.8
    assert coverage_score(dark_finger) > 0.8
    assert coverage_score(flash_finger) > 0.8
    assert coverage_score(scene) < 0.2
    assert coverage_score(face) < 0.2
    assert coverage_score(close_face) < 0.3


# ============================== estimate_pulse ==============================


def test_clean_signal_gives_accurate_bpm_and_good_quality():
    est = estimate_pulse(synth_ppg(72))
    assert est is not None
    assert abs(est.bpm - 72) < 1.0
    assert est.quality >= 55


def test_uses_real_timestamps_rather_than_assuming_30fps():
    samples = synth_ppg(72, fps=24)
    assert abs(legacy_bpm([s.r for s in samples]) - 72) > 10  # the old bug
    est = estimate_pulse(samples)
    assert abs(est.bpm - 72) < 1.5
    assert est.fps == pytest.approx(24, abs=0.5)


def test_dropped_frames_and_timing_jitter():
    est = estimate_pulse(
        synth_ppg(65, drop_frac=0.15, jitter_s=0.004, noise=0.5, seed=3)
    )
    assert abs(est.bpm - 65) < 2.0


def test_strong_second_harmonic_still_picks_fundamental():
    est = estimate_pulse(synth_ppg(60, harmonic=0.9))
    assert abs(est.bpm - 60) < 1.5


def test_exposure_step_mid_window():
    est = estimate_pulse(synth_ppg(80, step_at=6.0, step_size=25.0))
    assert abs(est.bpm - 80) < 2.0


def test_exposure_settling_transient_at_start():
    est = estimate_pulse(synth_ppg(70, settle=40.0))
    assert abs(est.bpm - 70) < 2.0


def test_baseline_wander_does_not_bias_rate():
    est = estimate_pulse(synth_ppg(58, drift=15.0, noise=0.4))
    assert abs(est.bpm - 58) < 1.5


def test_white_noise_is_poor_quality():
    est = estimate_pulse(synth_ppg(70, amp=0.0, noise=3.0))
    assert est is None or est.quality < 30


def test_noisy_signal_scores_between_poor_and_good():
    clean = estimate_pulse(synth_ppg(72)).quality
    noisy = estimate_pulse(synth_ppg(72, noise=2.5, seed=5)).quality
    none = estimate_pulse(synth_ppg(72, amp=0.0, noise=3.0))
    worst = none.quality if none else 0.0
    assert clean > noisy > worst


def test_saturated_red_falls_back_to_green_channel():
    est = estimate_pulse(synth_ppg(75, saturate="r", pulse_channel="g"))
    assert est.channel == "g"
    assert abs(est.bpm - 75) < 1.5


def test_equally_clean_channels_prefer_the_larger_pulse():
    # Noise-free: every channel has the same spectral share, only the pulse
    # amplitude differs (blue carries it). Order alone would pick red.
    est = estimate_pulse(
        synth_ppg(70, pulse_channel="b", noise=0.0, weak_channel_noise=0.0)
    )
    assert est.channel == "b"


def test_short_window_gives_a_reading():
    est = estimate_pulse(synth_ppg(72, duration_s=4.5))
    assert est is not None
    assert abs(est.bpm - 72) < 3.0


def test_too_short_returns_none():
    assert estimate_pulse(synth_ppg(72, duration_s=3.0)) is None


def test_too_sparse_returns_none():
    assert estimate_pulse(synth_ppg(72, duration_s=6.0, fps=6)) is None


@pytest.mark.parametrize("hr", [45, 55, 72, 90, 110, 140, 170])
def test_rate_sweep(hr):
    est = estimate_pulse(synth_ppg(hr, noise=0.3, seed=hr))
    assert abs(est.bpm - hr) < 1.5


def test_gap_in_data_uses_only_segment_after_gap():
    before = synth_ppg(72, duration_s=5.0)
    after = synth_ppg(72, duration_s=6.0)
    shifted = [
        FrameSample(s.t + 7.0, s.r, s.g, s.b, s.r_std, s.saturation) for s in after
    ]
    est = estimate_pulse(before + shifted)
    assert est is not None
    assert est.duration_s < 6.5
    assert abs(est.bpm - 72) < 2.0


def test_window_limits_analysis_to_recent_seconds():
    est = estimate_pulse(synth_ppg(72, duration_s=30.0), window_s=12.0)
    assert 11.5 <= est.duration_s <= 12.0


def test_estimate_exposes_waveform_and_spectrum_for_charts():
    est = estimate_pulse(synth_ppg(72))
    assert est.fps == pytest.approx(30, abs=0.5)
    assert len(est.filtered) == pytest.approx(12 * 30, abs=2)
    assert est.freqs_bpm.min() >= BAND_LOW_HZ * 60 - 1e-6
    assert est.freqs_bpm.max() <= BAND_HIGH_HZ * 60 + 1e-6
    assert len(est.freqs_bpm) == len(est.spectrum)
    assert est.n_beats >= 10  # 12 s at 72 bpm is 14 beats


def test_beat_intervals_agree_with_spectrum_on_clean_signal():
    est = estimate_pulse(synth_ppg(66))
    assert abs(est.ibi_bpm - 66) < 1.5
    assert abs(est.spectral_bpm - 66) < 1.5


def test_new_estimator_beats_legacy_on_a_condition_grid(capsys):
    """Benchmark: mean absolute error over realistic conditions."""
    conditions = {
        "clean 30fps": dict(),
        "24 fps": dict(fps=24),
        "20 fps": dict(fps=20),
        "15% drops": dict(drop_frac=0.15),
        "noise 1.5": dict(noise=1.5),
        "harmonic 0.9": dict(harmonic=0.9),
        "settle 40": dict(settle=40.0),
        "3 s reading": dict(duration_s=4.5),
    }
    rates = [52, 64, 75, 88, 105]
    rows = []
    for name, kw in conditions.items():
        err_new, err_old = [], []
        for hr in rates:
            samples = synth_ppg(hr, seed=hr, **kw)
            new = estimate_pulse(samples)
            old = legacy_bpm([s.r for s in samples])
            err_new.append(abs(new.bpm - hr) if new else 60.0)
            err_old.append(abs(old - hr) if old else 60.0)
        rows.append((name, float(np.mean(err_old)), float(np.mean(err_new))))

    with capsys.disabled():
        print("\n\nMAE in bpm (legacy → new), 5 rates per condition")
        for name, old, new in rows:
            print(f"  {name:<14} {old:6.2f} → {new:5.2f}")

    for name, old, new in rows:
        assert new <= old + 0.05, name
    assert np.mean([r[2] for r in rows]) < 1.0


# ============================== samples_from_video ==============================


def _synthetic_mp4(tmp_path, hr_bpm=72, duration_s=8.0, fps=25, amp=8.0):
    """Encode a uniform-colour clip whose red level pulses at hr_bpm."""
    path = tmp_path / "finger.mp4"
    n = int(duration_s * fps)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = 320, 240
        stream.pix_fmt = "yuv420p"
        stream.options = {"qscale": "1"}
        for i in range(n):
            red = 140 + amp * np.cos(2 * np.pi * hr_bpm / 60 * i / fps)
            img = np.empty((240, 320, 3), dtype=np.uint8)
            img[..., 0], img[..., 1], img[..., 2] = int(round(red)), 60, 40
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path.read_bytes()


def test_samples_from_video_recovers_rate_from_synthetic_clip(tmp_path):
    video = _synthetic_mp4(tmp_path, hr_bpm=72, duration_s=8.0, fps=25)
    samples = samples_from_video(video)
    assert 190 <= len(samples) <= 200  # 8 s at 25 fps
    assert np.allclose(np.diff([s.t for s in samples]), 1 / 25, atol=1e-6)
    assert all(s.r > s.g > s.b for s in samples)  # RGB order honoured
    est = estimate_pulse(samples)
    assert est is not None
    assert abs(est.bpm - 72) < 3.0


def test_samples_from_video_honours_max_seconds(tmp_path):
    video = _synthetic_mp4(tmp_path, duration_s=6.0, fps=25)
    samples = samples_from_video(video, max_seconds=2.0)
    assert samples[-1].t <= 2.0 + 1e-6
    assert len(samples) <= 51


# ============================== PulseSampler ==============================


def test_sampler_builds_timeline_from_pts():
    s = PulseSampler()
    for i in range(10):
        s.add_frame(red_frame(), pts_time=1000.0 + i / 30)
    snap = s.snapshot()
    assert len(snap) == 10
    assert np.allclose(np.diff([x.t for x in snap]), 1 / 30, atol=1e-6)


def test_sampler_falls_back_to_clock_when_pts_wraps_or_is_missing():
    clock = FakeClock(step=1 / 25)
    s = PulseSampler(clock=clock)
    s.add_frame(red_frame(), pts_time=47000.0)
    s.add_frame(red_frame(), pts_time=47000.0 + 1 / 30)
    s.add_frame(red_frame(), pts_time=5.0)  # RTP wrap: went backwards
    s.add_frame(red_frame(), pts_time=None)  # no pts at all
    ts = [x.t for x in s.snapshot()]
    assert ts[1] - ts[0] == pytest.approx(1 / 30, abs=1e-6)
    assert ts[2] - ts[1] == pytest.approx(1 / 25, abs=1e-6)
    assert ts[3] - ts[2] == pytest.approx(1 / 25, abs=1e-6)


def test_sampler_keeps_only_recent_seconds():
    s = PulseSampler(max_seconds=5.0)
    for i in range(300):
        s.add_frame(red_frame(), pts_time=i / 30)
    snap = s.snapshot()
    assert snap[-1].t - snap[0].t <= 5.0 + 1e-6
    assert len(snap) >= 140


def test_sampler_restarts_buffer_when_finger_is_placed():
    s = PulseSampler()
    for i in range(60):  # 2 s of uncovered scene
        s.add_frame(scene_frame(seed=i), pts_time=i / 30)
    for i in range(60, 90):  # 1 s covered
        s.add_frame(red_frame(), pts_time=i / 30)
    snap = s.snapshot()
    assert snap[-1].t - snap[0].t <= 1.0 + 1e-6
    assert all(coverage_score(x) > 0.5 for x in snap)


def test_sampler_ignores_brief_coverage_flicker():
    s = PulseSampler()
    frames = [red_frame()] * 90 + [scene_frame(seed=i) for i in range(6)] + [red_frame()] * 30
    for i, img in enumerate(frames):
        s.add_frame(img, pts_time=i / 30)
    snap = s.snapshot()
    assert snap[-1].t - snap[0].t == pytest.approx((len(frames) - 1) / 30, abs=1e-6)


def test_sampler_status_hints():
    s = PulseSampler()
    for i in range(15):
        s.add_frame(scene_frame(seed=i), pts_time=i / 30)
    assert s.status().hint == "uncovered"
    assert not s.status().covered

    s = PulseSampler()
    for i in range(15):
        s.add_frame(face_frame(), pts_time=i / 30)
    assert s.status().hint == "uncovered"
    assert not s.status().covered

    s = PulseSampler()
    for i in range(15):
        s.add_frame(red_frame(r=12, g=4, b=3), pts_time=i / 30)
    assert s.status().hint == "dark"

    s = PulseSampler()
    for i in range(15):
        s.add_frame(red_frame(r=255, g=120, b=60), pts_time=i / 30)
    assert s.status().hint == "saturated"

    s = PulseSampler()
    for i in range(15):
        s.add_frame(red_frame(), pts_time=i / 30)
    assert s.status().hint == "ok"
    assert s.status().covered
    assert s.status().coverage > 0.8


def test_sampler_status_empty_before_any_frame():
    status = PulseSampler().status()
    assert status.hint == "empty"
    assert not status.covered
    assert status.seconds == 0.0


def _av_frames(n, fps=30.0):
    frames = []
    for i in range(n):
        f = av.VideoFrame.from_ndarray(red_frame(), format="bgr24")
        f.pts = int(i * 90000 / fps)
        f.time_base = fractions.Fraction(1, 90000)
        frames.append(f)
    return frames


def test_recv_queued_samples_every_frame_and_returns_them_unchanged():
    s = PulseSampler()
    frames = _av_frames(7)
    out = asyncio.run(s.recv_queued(frames))
    assert out is frames
    snap = s.snapshot()
    assert len(snap) == 7
    assert np.allclose(np.diff([x.t for x in snap]), 1 / 30, atol=1e-6)


def test_recv_queued_counts_errors_instead_of_killing_the_stream():
    s = PulseSampler()
    frames = _av_frames(2) + [object()]
    out = asyncio.run(s.recv_queued(frames))
    assert len(out) == 3
    assert len(s.snapshot()) == 2
    assert s.errors == 1


def test_sampler_clear_empties_buffer_and_resets_timeline():
    s = PulseSampler()
    for i in range(30):
        s.add_frame(red_frame(), pts_time=i / 30)
    s.clear()
    assert s.snapshot() == []
    s.add_frame(red_frame(), pts_time=99.0)
    assert len(s.snapshot()) == 1


# ============================== PulseTracker ==============================


def test_tracker_smooths_a_single_outlier():
    tr = PulseTracker()
    state = None
    for i, (bpm, q) in enumerate([(72, 80), (73, 80), (110, 60), (72, 80), (71, 80)]):
        state = tr.update(i * 0.5, bpm=bpm, quality=q)
    assert abs(state.bpm - 72) < 1.5


def test_tracker_locks_after_consistent_acceptable_estimates():
    tr = PulseTracker()
    states = [tr.update(i * 0.5, bpm=70 + (i % 2), quality=70) for i in range(6)]
    assert not states[0].locked
    assert states[0].status == "acquiring"
    assert states[-1].locked
    assert states[-1].status == "stable"


def test_tracker_does_not_lock_on_poor_quality():
    tr = PulseTracker()
    states = [tr.update(i * 0.5, bpm=70, quality=20) for i in range(8)]
    assert not any(s.locked for s in states)
    assert states[-1].bpm is None
    assert states[-1].status == "searching"


def test_tracker_holds_then_searches_when_signal_is_lost():
    tr = PulseTracker()
    for i in range(6):
        state = tr.update(i * 0.5, bpm=70, quality=70)
    assert state.locked
    t0 = 3.0
    state = tr.update(t0, bpm=None, quality=0)
    assert state.status == "holding" and state.bpm == pytest.approx(70)
    assert not state.locked
    for k in range(1, 30):
        state = tr.update(t0 + k * 0.5, bpm=None, quality=0)
    assert state.status == "searching" and state.bpm is None


def test_tracker_reset_forgets_history():
    tr = PulseTracker()
    for i in range(6):
        tr.update(i * 0.5, bpm=70, quality=70)
    tr.reset()
    state = tr.update(10.0, bpm=90, quality=70)
    assert state.bpm == pytest.approx(90)
    assert not state.locked
