"""Photoplethysmography (PPG) pipeline for PulseIQ.

Pure numpy/scipy — no Streamlit imports — so it can be unit-tested and
benchmarked on synthetic recordings (tests/test_ppg.py).

    camera frame ──sample_frame──▶ FrameSample(t, r, g, b, r_std, saturation)
                                        │  PulseSampler: per-session buffer,
                                        │  real timestamps, finger detection
                                        ▼
    estimate_pulse: recent window ▶ resample to 30 Hz ▶ step removal, detrend,
                    band-pass ▶ Hann + zero-padded FFT ▶ harmonic-aware peak
                    ▶ best channel ▶ beat-interval confirmation ▶ quality
                                        │
                                        ▼
    PulseTracker: smooth successive estimates, decide when the reading is stable

Design note: the dominant error source for camera PPG is *timing*, not
amplitude. A 1–3 Hz pulse sampled at a nominal 30 fps that actually arrives at
24 fps is mis-read by 25% if the frame rate is assumed. Every sample therefore
carries a real capture timestamp and the analysis resamples onto a uniform grid
before any filtering.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

# ============================== Constants ==============================

# Passband. 0.7–3.0 Hz is 42–180 bpm: brackets bradycardia through marked
# tachycardia at rest while keeping the second harmonic of anything above
# 90 bpm out of band, which is where harmonic mis-picks come from.
BAND_LOW_HZ, BAND_HIGH_HZ = 0.7, 3.0

ANALYSIS_FS = 30.0  # uniform grid the raw frames are resampled onto (Hz)
MIN_WINDOW_S = 4.0  # shortest window that yields a reading
WINDOW_S = 12.0  # full analysis window
MAX_GAP_S = 1.0  # a gap longer than this restarts the window after it
MIN_SAMPLES_PER_S = 10.0  # below this the stream is too sparse to analyse
ROI_FRACTION = 0.5  # central fraction of width/height averaged per frame
FFT_N = 4096  # zero-padded FFT length (bin ≈ 0.44 bpm at 30 Hz)
QUALITY_BAND_HZ = 0.125  # half-width of the "signal" band around f0 and 2·f0
SUBHARMONIC_RATIO = 0.4  # f0/2 wins if it carries this share of the peak power
SATURATION_LEVEL = 250  # 8-bit level treated as clipped
CHANNELS = ("r", "g", "b")


# ============================== Frame sampling ==============================


@dataclass(frozen=True)
class FrameSample:
    """One camera frame reduced to the numbers the pipeline needs."""

    t: float  # capture time, seconds (arbitrary origin, monotonic)
    r: float  # mean red over the ROI
    g: float
    b: float
    r_std: float  # spatial std of red over the ROI (uniformity)
    saturation: float  # fraction of ROI red pixels at/above SATURATION_LEVEL


def _roi(img: np.ndarray, roi_fraction: float) -> np.ndarray:
    h, w = img.shape[:2]
    dy = int(h * (1.0 - roi_fraction) / 2.0)
    dx = int(w * (1.0 - roi_fraction) / 2.0)
    return img[dy : h - dy, dx : w - dx]


def sample_frame(
    img: np.ndarray,
    t: float,
    channel_order: str = "bgr",
    roi_fraction: float = ROI_FRACTION,
) -> FrameSample:
    """Reduce an 8-bit colour frame to a FrameSample over its centre ROI.

    `channel_order` is "bgr" for frames from av/OpenCV and "rgb" for imageio.
    """
    roi = _roi(img, roi_fraction)
    if channel_order == "bgr":
        b, g, r = roi[..., 0], roi[..., 1], roi[..., 2]
    elif channel_order == "rgb":
        r, g, b = roi[..., 0], roi[..., 1], roi[..., 2]
    else:
        raise ValueError(f"channel_order must be 'bgr' or 'rgb', got {channel_order!r}")
    return FrameSample(
        t=float(t),
        r=float(r.mean()),
        g=float(g.mean()),
        b=float(b.mean()),
        r_std=float(r.std()),
        saturation=float((r >= SATURATION_LEVEL).mean()),
    )


def coverage_score(sample: FrameSample) -> float:
    """Soft score in [0, 1] for "a fingertip is covering the lens".

    Light transmitted through a fingertip is strongly red-dominant (haemoglobin
    absorbs green and blue; R/G is typically 2–4) *and* spatially uniform. Lit
    skin seen from a distance is only mildly red-dominant (R/G ≈ 1.3–1.6) and a
    face is never uniform, so both factors are required and multiplied. A red
    channel clipped by a phone flash cannot be ratio-tested, so heavy
    saturation with non-white G/B counts as red-dominant.
    """
    ratio = sample.r / max(sample.g, sample.b, 1.0)
    dominance = float(np.clip((ratio - 1.4) / 0.8, 0.0, 1.0))
    if sample.saturation > 0.5 and max(sample.g, sample.b) < 200:
        dominance = 1.0
    cv = sample.r_std / max(sample.r, 1.0)
    uniformity = float(np.clip((0.35 - cv) / 0.20, 0.0, 1.0))
    return dominance * uniformity


def samples_from_video(video_bytes: bytes, max_seconds: float = 20.0) -> list[FrameSample]:
    """Decode a recording and reduce its first `max_seconds` to FrameSamples.

    Frame times come from the container's declared frame rate; imageio yields
    RGB frames. Frames that cannot be sampled are skipped rather than aborting
    the whole recording.
    """
    import imageio.v3 as iio  # heavy import, only needed for uploads

    try:
        meta = iio.immeta(video_bytes, plugin="pyav")
        fps = float(meta.get("fps") or 30.0)
    except Exception:
        fps = 30.0
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    samples: list[FrameSample] = []
    for index, frame in enumerate(iio.imiter(video_bytes, plugin="pyav")):
        t = index / fps
        if t > max_seconds:
            break
        if frame.ndim == 2:  # greyscale container: no colour to choose from
            frame = np.stack([frame] * 3, axis=-1)
        try:
            samples.append(sample_frame(frame[..., :3], t, channel_order="rgb"))
        except Exception:
            continue
    return samples


# ============================== Per-session sampler ==============================


@dataclass(frozen=True)
class SamplerStatus:
    coverage: float
    covered: bool  # a fingertip is on the lens: the only state worth estimating in
    brightness: float
    saturation: float
    hint: str  # "empty" | "uncovered" | "saturated" | "dark" | "ok"
    n_samples: int
    seconds: float


class PulseSampler:
    """Thread-safe buffer of FrameSamples with a stitched, monotonic timeline.

    One instance per browser session. `recv_queued` is the streamlit-webrtc
    callback: it samples *every* queued frame (the single-frame callback drops
    all but the newest when the worker falls behind) and returns the frames
    unchanged so the camera preview keeps working.

    Timeline: consecutive frame times are built from RTP presentation
    timestamps (`frame.time`) when the delta is sane, otherwise from the
    monotonic clock. This survives missing pts, RTP wrap-around and clock
    jumps without ever producing a non-increasing timeline.
    """

    MAX_PTS_DELTA_S = 2.0
    COVERED_ON, COVERED_OFF = 0.6, 0.3  # coverage EMA hysteresis
    UNCOVERED_RESET_S = 1.0  # min uncovered spell before placement restarts buffer
    STATUS_WINDOW_S = 1.0

    def __init__(
        self,
        max_seconds: float = 30.0,
        roi_fraction: float = ROI_FRACTION,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._lock = threading.Lock()
        self._samples: deque[FrameSample] = deque()
        self._max_seconds = max_seconds
        self._roi_fraction = roi_fraction
        self._clock = clock
        self.errors = 0
        self.frames = 0
        self._reset_timeline()

    def _reset_timeline(self) -> None:
        self._t: Optional[float] = None
        self._last_pts: Optional[float] = None
        self._last_clock: Optional[float] = None
        self._cov_ema: Optional[float] = None
        self._uncovered_since: Optional[float] = None

    # ---- ingestion ----

    def _next_time(self, pts_time: Optional[float], now: float) -> float:
        if self._t is None:
            t = now
        else:
            delta = None
            if pts_time is not None and self._last_pts is not None:
                d = pts_time - self._last_pts
                if 0.0 < d < self.MAX_PTS_DELTA_S:
                    delta = d
            if delta is None:
                delta = max(now - (self._last_clock or now), 1e-3)
            t = self._t + delta
        self._t, self._last_pts, self._last_clock = t, pts_time, now
        return t

    def _track_coverage(self, sample: FrameSample) -> None:
        cov = coverage_score(sample)
        self._cov_ema = cov if self._cov_ema is None else 0.8 * self._cov_ema + 0.2 * cov
        if self._cov_ema < self.COVERED_OFF:
            if self._uncovered_since is None:
                self._uncovered_since = sample.t
        elif self._cov_ema > self.COVERED_ON and self._uncovered_since is not None:
            if sample.t - self._uncovered_since >= self.UNCOVERED_RESET_S:
                # Finger just placed after a real absence: the frames before
                # this moment are the room, not the pulse.
                self._samples.clear()
            self._uncovered_since = None

    def add_frame(
        self, img: np.ndarray, pts_time: Optional[float] = None, channel_order: str = "bgr"
    ) -> FrameSample:
        now = self._clock()
        stats = sample_frame(img, 0.0, channel_order, self._roi_fraction)
        with self._lock:
            t = self._next_time(pts_time, now)
            sample = FrameSample(t, stats.r, stats.g, stats.b, stats.r_std, stats.saturation)
            self._track_coverage(sample)
            self._samples.append(sample)
            while self._samples and self._samples[0].t < t - self._max_seconds:
                self._samples.popleft()
            self.frames += 1
        return sample

    async def recv_queued(self, frames):
        """streamlit-webrtc `queued_video_frames_callback`."""
        for frame in frames:
            try:
                self.add_frame(frame.to_ndarray(format="bgr24"), pts_time=frame.time)
            except Exception:
                # Never let one bad frame kill the worker thread; the UI
                # surfaces the count.
                self.errors += 1
        return frames

    # ---- reading ----

    def snapshot(self) -> list[FrameSample]:
        with self._lock:
            return list(self._samples)

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()
            self._reset_timeline()

    def status(self) -> SamplerStatus:
        """Frame-level feedback for the last second: lens covered, exposure."""
        with self._lock:
            if not self._samples:
                return SamplerStatus(0.0, False, 0.0, 0.0, "empty", 0, 0.0)
            t_end = self._samples[-1].t
            recent = [s for s in self._samples if s.t >= t_end - self.STATUS_WINDOW_S]
            n = len(self._samples)
            seconds = t_end - self._samples[0].t
        coverage = float(np.mean([coverage_score(s) for s in recent]))
        brightness = float(np.mean([s.r for s in recent]))
        saturation = float(np.mean([s.saturation for s in recent]))
        covered = coverage >= self.COVERED_ON
        if not covered:
            hint = "uncovered"
        elif saturation > 0.5:
            hint = "saturated"
        elif brightness < 20.0:
            hint = "dark"
        else:
            hint = "ok"
        return SamplerStatus(coverage, covered, brightness, saturation, hint, n, seconds)


# ============================== Estimation ==============================


@dataclass
class PulseEstimate:
    bpm: float
    quality: float  # 0–100
    fps: float  # native frame rate of the analysed window
    duration_s: float
    channel: str
    spectral_bpm: float
    ibi_bpm: Optional[float]
    n_beats: int
    beat_consistency: float  # 0–1, regularity of inter-beat intervals
    spectral_share: float  # 0–1, in-band power at f0 and 2·f0
    filtered: np.ndarray  # z-scored band-passed waveform at ANALYSIS_FS
    freqs_bpm: np.ndarray  # in-band frequency axis for the chart
    spectrum: np.ndarray  # in-band power, normalised to max 1


_SOS = butter(4, [BAND_LOW_HZ, BAND_HIGH_HZ], btype="band", fs=ANALYSIS_FS, output="sos")


def _select_window(times: np.ndarray, window_s: float) -> np.ndarray:
    """Indices of the most recent `window_s` seconds, after the last long gap."""
    idx = np.nonzero(times >= times[-1] - window_s)[0]
    if len(idx) > 1:
        gaps = np.nonzero(np.diff(times[idx]) > MAX_GAP_S)[0]
        if len(gaps):
            idx = idx[gaps[-1] + 1 :]
    return idx


def _remove_steps(x: np.ndarray, k: float = 6.0) -> np.ndarray:
    """Cancel abrupt level shifts (auto-exposure jumps) without touching the pulse.

    A jump shows up as one outlying first difference. Replacing that
    difference by the median difference and integrating back leaves the
    waveform continuous while the pulse, whose per-sample changes are small,
    passes untouched.
    """
    d = np.diff(x)
    med = np.median(d)
    mad = 1.4826 * np.median(np.abs(d - med))
    if mad <= 0:
        return x
    jump = np.abs(d - med) > k * mad
    if not jump.any():
        return x
    d = d.copy()
    d[jump] = med
    return np.concatenate([[x[0]], x[0] + np.cumsum(d)])


def _condition(values: np.ndarray, rel_t: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Resample onto the grid, then step-remove, detrend, clip and band-pass."""
    x = np.interp(grid, rel_t, values)
    x = _remove_steps(x)
    x = x - np.polyval(np.polyfit(grid, x, 2), grid)
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    if mad > 0:
        x = np.clip(x, med - 4.0 * mad, med + 4.0 * mad)
    return sosfiltfilt(_SOS, x)


def _power_spectrum(y: np.ndarray):
    window = np.hanning(len(y))
    power = np.abs(np.fft.rfft(y * window, n=FFT_N)) ** 2
    freqs = np.fft.rfftfreq(FFT_N, d=1.0 / ANALYSIS_FS)
    band = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    return freqs[band], power[band]


def _parabolic_peak(freqs: np.ndarray, power: np.ndarray, k: int) -> float:
    if 0 < k < len(power) - 1:
        a, b, c = power[k - 1], power[k], power[k + 1]
        denom = a - 2.0 * b + c
        if denom < 0:
            shift = 0.5 * (a - c) / denom
            return float(freqs[k] + shift * (freqs[1] - freqs[0]))
    return float(freqs[k])


def _pick_fundamental(freqs: np.ndarray, power: np.ndarray) -> float:
    """In-band maximum, but prefer f/2 when the sub-harmonic is substantial.

    A pulse waveform is not a sine: its second harmonic can rival the
    fundamental, and picking it doubles the heart rate.
    """
    k = int(np.argmax(power))
    f0 = freqs[k]
    if f0 / 2.0 >= BAND_LOW_HZ:
        near = np.abs(freqs - f0 / 2.0) <= 0.06 * f0 / 2.0
        if near.any():
            j = int(np.argmax(np.where(near, power, -np.inf)))
            if power[j] >= SUBHARMONIC_RATIO * power[k]:
                k = j
    return _parabolic_peak(freqs, power, k)


def _spectral_share(freqs: np.ndarray, power: np.ndarray, f0: float) -> float:
    total = float(power.sum())
    if total <= 0:
        return 0.0
    signal = np.abs(freqs - f0) <= QUALITY_BAND_HZ
    if 2.0 * f0 <= BAND_HIGH_HZ:
        signal |= np.abs(freqs - 2.0 * f0) <= QUALITY_BAND_HZ
    return float(power[signal].sum() / total)


def _beat_intervals(y: np.ndarray, f0: float):
    """Median inter-beat rate with sub-sample peak refinement.

    Returns (bpm or None, n_beats, consistency in 0–1).
    """
    period = ANALYSIS_FS / f0
    peaks, _ = find_peaks(y, distance=max(1, int(0.6 * period)), prominence=0.5 * float(np.std(y)))
    if len(peaks) < 2:
        return None, len(peaks), 0.0
    refined = peaks.astype(float)
    for i, k in enumerate(peaks):
        if 0 < k < len(y) - 1:
            a, b, c = y[k - 1], y[k], y[k + 1]
            denom = a - 2.0 * b + c
            if denom < 0:
                refined[i] = k + 0.5 * (a - c) / denom
    ibi = np.diff(refined) / ANALYSIS_FS
    ibi = ibi[(ibi >= 0.6 / f0) & (ibi <= 1.5 / f0)]
    if len(ibi) < 2:
        return None, len(ibi) + 1, 0.0
    cv = float(np.std(ibi) / np.mean(ibi))
    consistency = float(np.clip(1.0 - cv / 0.3, 0.0, 1.0))
    return float(60.0 / np.median(ibi)), len(ibi) + 1, consistency


def estimate_pulse(
    samples: Sequence[FrameSample],
    *,
    window_s: float = WINDOW_S,
    min_window_s: float = MIN_WINDOW_S,
) -> Optional[PulseEstimate]:
    """Estimate heart rate from timestamped frame samples.

    Returns None when there is not yet enough usable data.
    """
    samples = sorted(samples, key=lambda s: s.t)
    if len(samples) < 2:
        return None
    times = np.array([s.t for s in samples], dtype=float)
    idx = _select_window(times, window_s)
    if len(idx) < 2:
        return None
    rel_t = times[idx] - times[idx[0]]
    duration = float(rel_t[-1])
    if duration < min_window_s:
        return None
    fps = (len(idx) - 1) / duration
    if fps < MIN_SAMPLES_PER_S:
        return None

    grid = np.arange(0.0, duration, 1.0 / ANALYSIS_FS)
    window = [samples[i] for i in idx]
    saturation = float(np.mean([s.saturation for s in window]))

    # Channel choice: highest spectral share wins. Sensor noise is the same in
    # every channel, so when two channels are equally clean the one with the
    # larger pulsatile amplitude has the better real SNR and breaks the tie.
    best = None
    for name in CHANNELS:
        raw = np.array([getattr(s, name) for s in window], dtype=float)
        if raw.std() <= 0:
            continue
        y = _condition(raw, rel_t, grid)
        freqs, power = _power_spectrum(y)
        if power.sum() <= 0:
            continue
        f0 = _pick_fundamental(freqs, power)
        share = _spectral_share(freqs, power, f0)
        score = share
        clipped = raw.mean() >= 0.97 * 255 or (name == "r" and saturation > 0.3)
        if clipped:
            score *= 0.5
        amplitude = float(np.std(y))
        candidate = (score, amplitude, name, y, freqs, power, f0, share)
        if best is None:
            best = candidate
        elif score > best[0] + 0.03 or (abs(score - best[0]) <= 0.03 and amplitude > 1.5 * best[1]):
            best = candidate
    if best is None:
        return None
    _, _, channel, y, freqs, power, f0, share = best

    spectral_bpm = 60.0 * f0
    ibi_bpm, n_beats, consistency = _beat_intervals(y, f0)

    bpm = spectral_bpm
    quality_cap = 100.0
    if ibi_bpm is not None and n_beats >= 4:
        disagreement = abs(ibi_bpm - spectral_bpm)
        if disagreement <= 8.0:
            bpm = 0.5 * (spectral_bpm + ibi_bpm)
        elif disagreement > 12.0:
            quality_cap = 29.0  # two methods disagree: not a trustworthy reading
        regularity = consistency
    else:
        regularity = 0.5  # too few beats to judge: neutral
    quality = min(quality_cap, 100.0 * share * (0.5 + 0.5 * regularity))

    y_std = float(np.std(y))
    filtered = (y - y.mean()) / y_std if y_std > 0 else y - y.mean()
    return PulseEstimate(
        bpm=float(bpm),
        quality=float(quality),
        fps=float(fps),
        duration_s=duration,
        channel=channel,
        spectral_bpm=float(spectral_bpm),
        ibi_bpm=ibi_bpm,
        n_beats=int(n_beats),
        beat_consistency=float(consistency),
        spectral_share=float(share),
        filtered=filtered,
        freqs_bpm=freqs * 60.0,
        spectrum=power / power.max(),
    )


# ============================== Tracking ==============================


@dataclass(frozen=True)
class TrackerState:
    bpm: Optional[float]
    quality: float
    locked: bool
    status: str  # "searching" | "acquiring" | "stable" | "holding"


def _weighted_median(values: Iterable[float], weights: Iterable[float]) -> float:
    v = np.asarray(list(values), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    return float(v[int(np.searchsorted(cum, 0.5 * cum[-1]))])


class PulseTracker:
    """Turn a stream of noisy per-window estimates into a monitor-style readout.

    The displayed value is the quality-weighted median of acceptable
    estimates from the last `smooth_s` seconds. A reading is `locked` once
    the last `lock_n` estimates are all acceptable and agree within
    `lock_tol_bpm`. If the signal is lost the last value is held for
    `hold_s` seconds before the readout goes to standby.
    """

    def __init__(
        self,
        history_s: float = 12.0,
        smooth_s: float = 6.0,
        hold_s: float = 5.0,
        lock_n: int = 4,
        lock_tol_bpm: float = 4.0,
        accept_quality: float = 30.0,
    ):
        self.history_s = history_s
        self.smooth_s = smooth_s
        self.hold_s = hold_s
        self.lock_n = lock_n
        self.lock_tol_bpm = lock_tol_bpm
        self.accept_quality = accept_quality
        self.reset()

    def reset(self) -> None:
        self._hist: deque[tuple[float, Optional[float], float]] = deque()
        self._last_value: Optional[float] = None
        self._last_good_t: Optional[float] = None

    def _acceptable(self, bpm: Optional[float], quality: float) -> bool:
        return bpm is not None and quality >= self.accept_quality

    def update(self, t: float, bpm: Optional[float] = None, quality: float = 0.0) -> TrackerState:
        self._hist.append((t, bpm, quality))
        while self._hist and self._hist[0][0] < t - self.history_s:
            self._hist.popleft()

        if not self._acceptable(bpm, quality):
            if self._last_value is not None and t - self._last_good_t <= self.hold_s:
                return TrackerState(self._last_value, quality, False, "holding")
            return TrackerState(None, quality, False, "searching")

        recent = [(b, q) for tt, b, q in self._hist if tt >= t - self.smooth_s and self._acceptable(b, q)]
        value = _weighted_median((b for b, _ in recent), (q for _, q in recent))
        self._last_value, self._last_good_t = value, t

        tail = list(self._hist)[-self.lock_n :]
        locked = len(tail) == self.lock_n and all(
            self._acceptable(b, q) and abs(b - value) <= self.lock_tol_bpm for _, b, q in tail
        )
        return TrackerState(value, quality, locked, "stable" if locked else "acquiring")
