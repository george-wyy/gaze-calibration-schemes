"""Synthetic dragging sessions.

The library ships no participant data. These synthetic sessions exist so that every
scheme can be exercised end to end -- by a reader, by CI, and by anyone auditing the
code -- without a recording.

The generator produces repeated left-button drags over a Fitts-style grid of distances and
target widths, at 90 Hz on a 1920x1080 display. What it models:

* **Cursor**: a minimum-jerk reach from the press point to the target centre, preceded
  by a settle and followed by a hold, so that press and release are separated by a
  real trajectory.
* **Gaze**: a predictive copy of the cursor, advanced by a lead that is large at
  movement onset and decays to a small steady value -- the eye reaches for the target
  first and then tracks the cursor -- plus Gaussian noise.

This is a *model*, not a recording: it reproduces the qualitative structure the schemes
key on (correlated pursuit, stable fixations before release, a target entry test) and
nothing more. Numbers obtained from it say nothing about human performance.
"""

from __future__ import annotations

import numpy as np

from .session import Drag, Session

#: Movement time, `a + b * log2(2D/W)`. [here] -- chosen so that the synthetic reaches
#: span a plausible range; not fitted to any data.
FITTS_A_S = 0.20
FITTS_B_S = 0.15

#: Per-drag phases, in seconds. [here]
PRE_PRESS_S = 0.35
HOLD_BEFORE_RELEASE_S = 0.45
INTER_DRAG_GAP_S = 0.35

#: Gaze model. [here] -- the steady lead is of the order measured for gaze during a drag;
#: the onset lead and its decay are modelling choices. They are present because a real
#: drag begins with a large gaze-cursor offset: without them a fixture whose gaze simply
#: shadows the cursor with a 6 ms lead leaves Smooth-i with nothing to keep, since that
#: scheme retains only pairs beyond a 1.5 degree offset.
#:
#: The noise level is set so that a fixation is actually a fixation: the expected
#: dispersion of a nine-frame fixation is about 5.9 sigma, so at 0.13 degrees of noise
#: it lands near 0.8 degrees, inside the 1 degree I-DT gate that Sidenmark et al. use.
#: A noisier synthetic eye would make that published threshold unsatisfiable and every
#: fixation detector in this package would silently return nothing.
GAZE_LEAD_S = 0.006
GAZE_LEAD_ONSET_S = 0.25
GAZE_LEAD_ONSET_FRACTION = 0.6
GAZE_LEAD_DECAY_S = 0.18
GAZE_NOISE_PX = 5.0


def _minimum_jerk(tau: np.ndarray) -> np.ndarray:
    """Normalised minimum-jerk displacement profile on ``tau`` in ``[0, 1]``."""
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def _fitts_time(distance_px: float, width_px: float) -> float:
    return FITTS_A_S + FITTS_B_S * float(np.log2(2.0 * distance_px / width_px))


def session(
    distance_px: float = 534.0,
    width_px: float = 64.0,
    n_drags: int = 3,
    seed: int = 0,
    fs: float = 90.0,
) -> Session:
    """A synthetic session of ``n_drags`` back-and-forth drags.

    ``distance_px`` and ``width_px`` are the nominal Fitts parameters; drags alternate
    direction so that the cursor path is not monotone. ``seed`` fixes the gaze noise.
    """
    if distance_px <= 0 or width_px <= 0:
        raise ValueError("distance_px and width_px must be positive")
    if n_drags < 1:
        raise ValueError("n_drags must be at least 1")

    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    move_s = _fitts_time(distance_px, width_px)

    centre = np.array([960.0, 508.0])
    half = distance_px / 2.0
    # Drags run along radial directions spread over half a turn, offset by 45 degrees so
    # that no drag is axis-aligned. The offset is deliberate: the pursuit-based schemes
    # require correlation on *both* axes, and the correlation of a constant signal is
    # undefined, so a purely horizontal drag can never satisfy them. Real pointing tasks
    # are not axis-aligned either.
    angles = np.pi / 4.0 + np.linspace(0.0, np.pi, n_drags, endpoint=False)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    starts = [centre - d * half for d in directions]
    targets = [centre + d * half for d in directions]

    # ------------------------------------------------------------- assemble frames
    frames: list[dict] = []
    spans: list[tuple[int, int, np.ndarray]] = []

    def emit(count: int, cursor: np.ndarray) -> tuple[int, int]:
        first = len(frames)
        for c in cursor:
            frames.append({"cursor": c})
        return first, len(frames) - 1

    for k in range(n_drags):
        start, target = starts[k], targets[k]
        # settle before the press
        settle = np.repeat(start[None, :], int(round(PRE_PRESS_S * fs)), axis=0)
        pre_first, pre_last = emit(len(settle), settle)
        press = pre_last + 1
        # movement
        n_move = max(3, int(round(move_s * fs)))
        tau = np.linspace(0.0, 1.0, n_move)
        path = start[None, :] + (target - start)[None, :] * _minimum_jerk(tau)[:, None]
        emit(len(path), path)
        # hold at the target
        hold = np.repeat(target[None, :], int(round(HOLD_BEFORE_RELEASE_S * fs)), axis=0)
        hold_first, hold_last = emit(len(hold), hold)
        release = hold_last
        spans.append((press, release, target))
        # gap
        gap = np.repeat(target[None, :], int(round(INTER_DRAG_GAP_S * fs)), axis=0)
        emit(len(gap), gap)
        _ = pre_first

    n = len(frames)
    t = np.arange(n, dtype=float) * dt
    cursor = np.array([f["cursor"] for f in frames], dtype=float)

    # ------------------------------------------------------------------ gaze model
    # A predictive copy of the cursor. The lead is large at movement onset and decays
    # towards GAZE_LEAD_S, so early in a drag the gaze runs well ahead of the cursor:
    # the eye reaches for the target before the hand arrives. The decay is anchored at
    # each press and runs to the next one, which keeps the lead short through the
    # settle and hold phases.
    lead_frames = np.full(n, GAZE_LEAD_S / dt, dtype=float)
    press_times = [p for p, _r, _t in spans]
    # The onset lead is capped by a fraction of the movement time. On a short, wide
    # drag a fixed 250 ms lead would place the gaze at the target before the movement
    # began, pinning it there and destroying the correlation the pursuit schemes key
    # on. Scaling it keeps the peak gaze-cursor offset a roughly constant fraction of
    # the drag distance across the whole grid.
    onset_lead = min(GAZE_LEAD_ONSET_S, GAZE_LEAD_ONSET_FRACTION * move_s)
    for k, press in enumerate(press_times):
        stop = press_times[k + 1] if k + 1 < len(press_times) else n
        if stop <= press:
            continue
        elapsed = np.arange(stop - press, dtype=float) * dt
        lead = GAZE_LEAD_S + (onset_lead - GAZE_LEAD_S) * np.exp(-elapsed / GAZE_LEAD_DECAY_S)
        lead_frames[press:stop] = lead / dt

    idx = np.clip(np.arange(n, dtype=float) + lead_frames, 0.0, n - 1.0)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, n - 1)
    frac = (idx - lo)[:, None]
    gaze = cursor[lo] * (1.0 - frac) + cursor[hi] * frac
    gaze = gaze + rng.normal(0.0, GAZE_NOISE_PX, size=gaze.shape)

    # ------------------------------------------- target geometry for Online-EYE
    target_centre = np.repeat(targets[0][None, :], n, axis=0)
    for k, (press, release, target) in enumerate(spans):
        next_press = spans[k + 1][0] if k + 1 < len(spans) else n
        target_centre[press:next_press] = target[None, :]
    target_distance = np.hypot(cursor[:, 0] - target_centre[:, 0], cursor[:, 1] - target_centre[:, 1])

    return Session(
        t=t,
        gaze_x=gaze[:, 0],
        gaze_y=gaze[:, 1],
        cursor_x=cursor[:, 0],
        cursor_y=cursor[:, 1],
        drags=[Drag(press=p, release=r) for p, r, _ in spans],
        target_width_px=float(width_px),
        target_distance_px=target_distance,
    ).validate()


#: The nine distance x width cells used by :func:`grid_sessions` and the examples.
STUDY2_GRID: tuple[tuple[float, float], ...] = tuple(
    (d, w) for d in (150.0, 534.0, 918.0) for w in (32.0, 64.0, 128.0)
)


def grid_sessions(n_drags: int = 2, seed: int = 0) -> dict[tuple[float, float], Session]:
    """One session per cell of :data:`STUDY2_GRID`, keyed by ``(distance, width)``."""
    return {
        (d, w): session(distance_px=d, width_px=w, n_drags=n_drags, seed=seed + i)
        for i, (d, w) in enumerate(STUDY2_GRID)
    }
