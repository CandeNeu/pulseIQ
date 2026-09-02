# Live camera pulse reading quality — design

Date: 2026-09-02
Scope: the "Live camera" acquisition mode in `frontend/app.py` and the signal
pipeline it shares with "Upload recording".

Note: this design was produced in an autonomous session, so it was not
approved interactively before implementation. Decisions the author should
review are listed under "Tunables" at the end.

## Problem

The live heart-rate reading is inaccurate and jittery. Root causes found in
`frontend/app.py`:

1. Sample rate is assumed to be 30 fps. WebRTC delivers a variable rate
   (typically 15–30 fps depending on device and load) and no timestamps are
   kept, so any deviation maps directly into a heart-rate error.
2. With `async_processing=True`, streamlit-webrtc drops all but the newest
   queued frame unless the processor implements `recv_queued`.
3. The spectrum is an un-windowed, un-padded FFT: 6 bpm bins on 10 s of data,
   20 bpm bins on the first 3 s reading.
4. The signal is the red-channel mean of the whole frame. Edge light leakage
   and red-channel saturation (phone flash) both hurt; the green channel often
   carries the pulse when red is clipped.
5. No temporal smoothing or lock logic; the accepted value is whatever was on
   screen last, even if the finger had just moved.
6. Per-user buffer and lock are module globals rebound on every script run
   (shared across sessions).

## Goals

- Heart-rate error under ±2 bpm on a clean signal, independent of the actual
  camera frame rate and of dropped frames.
- Correct fundamental when the waveform has a strong second harmonic.
- A stable displayed value that only becomes "accepted" once it has held
  steady with acceptable quality.
- Actionable feedback: lens not covered, too dark, saturated, hold still.
- Pure-Python pipeline that is unit-tested on synthetic signals.

Non-goals: face (non-contact) rPPG, torch/exposure control from the browser,
multi-patch ROI selection.

## Architecture

```
frontend/
  ppg.py      # pure numpy/scipy pipeline (no Streamlit)
  app.py      # UI; owns a PulseSampler + PulseTracker per session
tests/
  test_ppg.py       # synthetic-signal tests and legacy-vs-new benchmark
  test_app_smoke.py # AppTest: the app renders through step 2
```

### `frontend/ppg.py`

**`FrameSample`** — one camera frame reduced to `(t, r, g, b, r_std,
saturation)`: timestamp in seconds, per-channel means over the centre ROI
(central 50% of width and height), spatial std of red, fraction of
near-saturated red pixels.

**`sample_frame(img, t, channel_order)`** — builds a `FrameSample`. Used by
the live callback (BGR from `av`) and the upload extractor (RGB from imageio).

**`coverage_score(sample) -> [0, 1]`** — soft "finger is on the lens" score:
red dominance (R relative to max(G, B), ramp 1.4 → 2.2; a flash-saturated
red counts as dominant) times spatial uniformity (R std / R mean, ramp
0.35 → 0.15). Lit skin seen from a distance has R/G ≈ 1.3–1.6 and a face is
never uniform, so a face scores ≈ 0 while a fingertip scores ≈ 1.
`PulseSampler.status().covered` is true when the last second averages ≥ 0.6.
The live monitor estimates nothing while the lens is not covered — a face
or a room is not a plethysmograph, and showing a number for it was the most
visible failure of the previous version.

**`PulseSampler`** — per-session, thread-safe ring buffer of `FrameSample`s
(last 30 s). `recv_queued(frames)` is the async streamlit-webrtc callback:
it samples every queued frame (nothing dropped) and returns the frames
unchanged so the preview still shows. Timeline policy: consecutive frame
times are stitched from RTP presentation timestamps (`frame.time`) when the
delta is sane (0 < Δ < 2 s), otherwise from the monotonic clock. This
survives missing `pts`, RTP wrap-around and clock jumps. On a rising edge of
coverage (uncovered for ≥ 1 s, then covered) the buffer is cleared so the
pre-placement frames do not pollute the window.

**`estimate_pulse(samples, window_s=12, min_window_s=4) -> PulseEstimate | None`**

1. Keep the most recent `window_s` seconds; if a gap > 1 s exists, keep only
   the data after the last gap. Require ≥ `min_window_s` seconds and ≥ 10
   samples per second on average.
2. Native fps = (n − 1) / span. Resample each channel onto a uniform 30 Hz
   grid by linear interpolation.
3. Per channel: subtract a quadratic trend, clip outliers at 4 MAD, band-pass
   0.7–3.0 Hz (42–180 bpm) with a 4th-order Butterworth `sosfiltfilt`,
   Hann window, 4096-point rFFT, in-band power.
4. Fundamental picking: take the in-band maximum; if a sub-harmonic at f/2
   carries ≥ 40% of that peak's power, choose f/2. Refine the frequency with
   parabolic interpolation over the three bins around the peak.
5. Channel choice: the channel with the highest spectral share (below); a
   channel whose mean is within 3% of full scale is penalised ×0.5.
6. Beat confirmation on the chosen channel: `find_peaks` with minimum
   distance 0.6 × period and prominence ≥ 0.5 × std; inter-beat intervals
   outside 0.6–1.5 × period are rejected. Median interval → `ibi_bpm`;
   coefficient of variation → `beat_consistency`.
7. Fusion: `bpm = spectral_bpm`; if ≥ 4 beats and |ibi − spectral| ≤ 8 bpm,
   `bpm = mean(spectral, ibi)`. If ≥ 4 beats and they disagree by > 12 bpm
   the reading is ambiguous and quality is capped at 29 ("Poor").
8. Quality (0–100) = share of in-band power within ±0.125 Hz of the
   fundamental plus ±0.125 Hz of its second harmonic (when in band). The
   band is fixed, not window-adaptive, so short windows honestly score lower
   and quality rises as the window fills. Expected: white noise ≈ 20%,
   clean 12 s signal > 85%.

Returns the estimate with the filtered 30 Hz waveform (chosen channel,
z-scored) and the in-band spectrum for the charts, plus diagnostics
(channel, native fps, duration, spectral/IBI bpm, beat count).

**`PulseTracker`** — smooths successive estimates for display and decides
when a reading is accepted. Keeps (t, bpm, quality) for 12 s. Displayed bpm
is the quality-weighted median of acceptable (≥ 30%) estimates from the last
6 s; if none, the previous value is held for 5 s ("holding"), then the
readout goes to standby ("searching"). `locked` when the last 4 estimates
are acceptable and within ±4 bpm of the displayed value.

### `frontend/app.py`

- `PulseSampler` and `PulseTracker` are created once per session in
  `st.session_state`; the module-level lock/deque and `video_frame_callback`
  are removed.
- `webrtc_streamer(..., queued_video_frames_callback=sampler.recv_queued)`
  with media constraints preferring the rear camera (`facingMode:
  environment`, the one with the flash on phones), 640×480 and 30 fps.
- The live fragment (every 0.5 s) reads the sampler status; only when the
  lens is covered does it run `estimate_pulse`, then it updates the tracker
  and renders. The HR readout shows the tracker's smoothed value, or "- -"
  until a window has passed the quality gate, with a badge of Acquiring /
  Steady / Holding / Searching / Captured; the charts show the current
  window. One guidance line says what to do next ("Lens not covered", "Too
  dark", "Sensor saturated", "Hold still", "Signal lost", "Reading
  captured") and a progress bar shows how much steady signal has been seen.
- A reading is **captured** when the tracker is locked *and* the window is
  full (≥ 11.5 s) *and* at least 13.5 s of signal have been buffered since
  the finger was placed, so the exposure transient never sits inside the
  accepted window. At that moment `measured_bpm`, `signal_pct`, `source` and
  an `analysis` record are stored and one full `st.rerun()` enables the "Run
  assessment" button (it lives outside the fragment). While captured, the
  stored values track the smoothed reading.
- The analysis record (window length, camera fps, channel, spectral rate,
  beat-interval rate and beat count, regularity, quality, capture time) is
  rendered as an "Analysis summary" table with a green "Reading accepted"
  banner, in both modes, and persists across reruns and when the camera is
  stopped. It also goes into the exported report. A "Restart reading" button
  discards it.
- The WebRTC component is created inside a guard: if it cannot start, the
  page shows an error panel and the idle monitor instead of a stack trace.
- Buffers are cleared on "Begin acquisition", "Back", "New session",
  "Restart reading", and whenever the stream transitions from stopped to
  playing.
- The upload path reuses `samples_from_video` + `estimate_pulse`: up to 20 s
  are decoded and the estimator analyses the most recent 12 s, which skips
  the exposure settling at the start of a phone recording. A rejected upload
  clears any previously accepted reading. `render_trace` is render-only.
- Sidebar and report read the passband and window from `ppg`.

## Error handling

- Estimator returns `None` for too little data or a degenerate spectrum; the
  UI shows the acquiring state with a sample-count progress.
- Any exception inside the frame callback is caught and counted (the
  streamlit-webrtc worker would otherwise die silently); the UI reports it.
- Timestamp anomalies fall back to the monotonic clock rather than failing.

## Testing

Synthetic PPG generator: sum of fundamental + second harmonic (dicrotic
notch), baseline wander, exposure-settling exponential, white noise, random
dropped frames, timing jitter, optional saturation. Tests assert:

- clean 72 bpm at 30 fps → within ±1 bpm, quality ≥ 55
- 24 fps stream with correct timestamps → within ±1.5 bpm (legacy gives ~90)
- 15% dropped frames + jitter → within ±2 bpm
- strong second harmonic → fundamental chosen
- exposure step / settling → within ±2 bpm
- white noise → `None` or quality < 30
- saturated red with pulse in green → green chosen, correct bpm
- 4 s window → within ±3 bpm
- 45–170 bpm sweep → within ±1.5 bpm
- tracker: outlier rejection, lock/hold/search transitions
- sampler: pts stitching, wrap fallback, coverage reset, face/scene → not
  covered, dark / saturated / ok hints
- upload decode: a synthetic mp4 (25 fps, 72 bpm, encoded with mpeg4) is
  decoded and read back within ±3 bpm
- benchmark: new MAE < legacy MAE over a condition grid (printed with -s)

AppTest smoke: step 1 renders; "Begin acquisition" creates the per-session
sampler and tracker; step 2 starts idle with "Run assessment" disabled; live
mode renders (error panel + idle monitor, since the component cannot start
headless); a seeded captured reading renders the analysis summary and the
accepted banner, enables "Run assessment", and clicking it reaches step 3.

Verified once by hand in a real browser (Playwright): uploading the synthetic
clip produced 72 bpm at 82% quality, spectral 71.9 vs beat-interval 71.8 bpm
over 15 beats, and the enabled assessment button.

## Tunables (for the author to review)

| Constant | Value | Where |
|---|---|---|
| Passband | 0.7–3.0 Hz (42–180 bpm) | `ppg.BAND_LOW_HZ/BAND_HIGH_HZ` |
| Analysis window | 4 s minimum, 12 s full | `ppg.MIN_WINDOW_S/WINDOW_S` |
| Quality thresholds | Good ≥ 55, Fair ≥ 30 | `app.grade_quality` |
| Lock rule | 4 consecutive acceptable within ±4 bpm | `ppg.PulseTracker` |
| Capture rule | locked + window ≥ 11.5 s + 13.5 s buffered | `app.reading_captured` |
| Coverage heuristic ramps | red ratio 1.4→2.2, CV 0.35→0.15, covered ≥ 0.6 | `ppg.coverage_score`, `PulseSampler.COVERED_ON` |
| ROI | central 50% × 50% | `ppg.ROI_FRACTION` |
