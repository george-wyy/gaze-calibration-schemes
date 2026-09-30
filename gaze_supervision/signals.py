"""Small signal utilities shared by the schemes.

These operate on plain numpy arrays so that nothing here depends on how the data was
recorded: a caller supplies timestamps in seconds and positions in screen pixels.
"""

from __future__ import annotations

import numpy as np


def dispersion(x: np.ndarray, y: np.ndarray) -> float:
    """Sum of the x and y ranges -- the dispersion measure used by I-DT here."""
    if len(x) == 0:
        return float("inf")
    return float((x.max() - x.min()) + (y.max() - y.min()))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, defined as 0.0 when either signal is constant."""
    if len(a) < 3:
        return 0.0
    sa, sb = float(a.std()), float(b.std())
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def idt_fixations(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    lo: int,
    hi: int,
    min_duration_s: float,
    dispersion_threshold_px: float,
) -> list[tuple[int, int]]:
    """Identify-dispersion-threshold fixation detection on the frame window ``[lo, hi)``.

    Returns inclusive ``(first, last)`` frame index pairs for each fixation found.
    ``hi`` is exclusive, matching a half-open slice.
    """
    out: list[tuple[int, int]] = []
    i = lo
    while i < hi:
        j = i
        while j + 1 < hi and (t[j] - t[i]) < min_duration_s:
            j += 1
        if (t[j] - t[i]) < min_duration_s:
            break
        if dispersion(x[i : j + 1], y[i : j + 1]) > dispersion_threshold_px:
            i += 1
            continue
        while j + 1 < hi and dispersion(x[i : j + 2], y[i : j + 2]) <= dispersion_threshold_px:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def smoothed_speed(x: np.ndarray, y: np.ndarray, t: np.ndarray, smooth_frames: int = 3) -> np.ndarray:
    """Gaze speed in px/s, averaged over ``smooth_frames`` adjacent frames.

    Frame-wise speed is unusable at 90 Hz on this display: consecutive frames repeat
    often enough that the raw median speed is zero. Averaging over three frames is the
    registered realisation (see docs/DEVIATIONS.md).
    """
    if len(x) < 2:
        return np.zeros(len(x), dtype=float)
    dt = np.diff(t)
    dt[dt <= 0] = np.finfo(float).eps
    step = np.hypot(np.diff(x), np.diff(y)) / dt
    speed = np.empty(len(x), dtype=float)
    speed[0] = step[0]
    speed[1:] = step
    if smooth_frames > 1:
        kernel = np.ones(smooth_frames, dtype=float) / smooth_frames
        speed = np.convolve(speed, kernel, mode="same")
    return speed


def rms_radius(x: np.ndarray, y: np.ndarray) -> float:
    """Root-mean-square radius of a point cloud about its own centroid, in pixels."""
    if len(x) == 0:
        return 0.0
    dx = x - x.mean()
    dy = y - y.mean()
    return float(np.sqrt(np.mean(dx * dx + dy * dy)))
