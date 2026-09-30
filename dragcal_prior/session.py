"""The data contract every scheme consumes.

A :class:`Session` is one recording unit -- for example, one (D, W) block by one repetition
of a pointing task. Nothing here is specific to any one apparatus: supply timestamps in
seconds and positions in screen pixels and the schemes run.

The contract is deliberately narrow. A scheme is a pure function
``Session -> list[Pair]``; it must not read files, and it must not assume that a
participant identifier exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import constants


@dataclass(frozen=True)
class Drag:
    """One button-down / button-up interval, as inclusive frame indices."""

    press: int
    release: int

    def __post_init__(self) -> None:
        if self.press < 0 or self.release < self.press:
            raise ValueError(f"invalid drag span: press={self.press} release={self.release}")


@dataclass(frozen=True)
class Pair:
    """One supervision pair: what the gaze was, and what the reference position was.

    ``frame`` is the index in the session that the pair is anchored to and ``t`` its
    timestamp; schemes differ in which of the two they anchor to, which is exactly the
    ``tau`` column of the comparison table.
    """

    gaze_x: float
    gaze_y: float
    reference_x: float
    reference_y: float
    frame: int
    t: float


@dataclass
class Session:
    """One recording unit of a dragging task.

    Parameters
    ----------
    t:
        Frame timestamps in seconds, strictly increasing, shape ``(n,)``.
    gaze_x, gaze_y:
        Gaze position in screen pixels, shape ``(n,)``.
    cursor_x, cursor_y:
        Cursor position in screen pixels, shape ``(n,)``. The uninstructed cursor path,
        which stands in for the display stimulus that the pursuit-based schemes assume.
    drags:
        The button-down / button-up intervals in this unit.
    px_per_degree:
        Display scale, used to realise the angular thresholds of the source papers.
    target_width_px:
        Target width, needed by Online-EYE (its ``W/2`` entry test). ``None`` disables
        that scheme with a clear error rather than guessing a value.
    target_distance_px:
        Distance in pixels from the cursor to the centre of the target the current drag
        is heading for, shape ``(n,)``. Needed by Online-EYE.
    screen_width_px, screen_height_px:
        Display bounds, used to normalise coordinates for the polynomial model.
    gaze_speed_smoothed_px_s:
        Optional precomputed gaze speed. Computed on demand if omitted.
    gaze_rms_px:
        Optional gaze noise estimate, used to set Online-EYE's I-DT dispersion
        threshold. Computed from consecutive-frame differences if omitted.
    """

    t: np.ndarray
    gaze_x: np.ndarray
    gaze_y: np.ndarray
    cursor_x: np.ndarray
    cursor_y: np.ndarray
    drags: list[Drag]
    px_per_degree: float = constants.PX_PER_DEGREE
    target_width_px: float | None = None
    target_distance_px: np.ndarray | None = None
    screen_width_px: float = constants.SCREEN_WIDTH_PX
    screen_height_px: float = constants.SCREEN_HEIGHT_PX
    gaze_speed_smoothed_px_s: np.ndarray | None = None
    gaze_rms_px: float | None = None
    _speed_cache: np.ndarray | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ basics --
    @property
    def n(self) -> int:
        return int(len(self.t))

    def validate(self) -> "Session":
        """Check shapes and monotonicity. Raises ``ValueError`` on any violation."""
        arrays = {
            "gaze_x": self.gaze_x,
            "gaze_y": self.gaze_y,
            "cursor_x": self.cursor_x,
            "cursor_y": self.cursor_y,
        }
        for name, arr in arrays.items():
            arr = np.asarray(arr, dtype=float)
            if arr.ndim != 1 or len(arr) != self.n:
                raise ValueError(f"{name} must be 1-D with the same length as t")
            arrays[name] = arr
        if self.n < 2:
            raise ValueError("a session needs at least two frames")
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        for d in self.drags:
            if d.release >= self.n:
                raise ValueError(f"drag {d} reaches past the last frame ({self.n - 1})")
        if self.target_distance_px is not None and len(self.target_distance_px) != self.n:
            raise ValueError("target_distance_px must have the same length as t")
        return self

    # -------------------------------------------------------------- accessors --
    def speed(self) -> np.ndarray:
        """Gaze speed in px/s, computed once and cached."""
        if self.gaze_speed_smoothed_px_s is not None:
            return self.gaze_speed_smoothed_px_s
        if self._speed_cache is None:
            from .signals import smoothed_speed

            self._speed_cache = smoothed_speed(self.gaze_x, self.gaze_y, self.t)
        return self._speed_cache

    def noise_radius_px(self) -> float:
        """Robust gaze-noise estimate in pixels.

        The fallback is the *median* consecutive-frame gaze displacement. A mean or an
        RMS would not do: a session spends much of its time in pursuit, and the moving
        frames dominate an average, which then inflates the I-DT dispersion threshold
        that Online-EYE derives from this value. The median is set by the quiet frames.

        Supply ``gaze_rms_px`` when you have a dedicated noise estimate. Matching a
        published number requires the same estimate that study used; this fallback is a
        documented approximation, not that estimate.
        """
        if self.gaze_rms_px is not None:
            return float(self.gaze_rms_px)
        step = np.hypot(np.diff(self.gaze_x), np.diff(self.gaze_y))
        if len(step) == 0:
            return 0.0
        return float(np.median(step))

    def events(self) -> list[tuple[str, int, int]]:
        """All press and release events as ``(kind, frame, drag_index)``, time-ordered.

        Every event-anchored scheme below is driven by this list. Mapping a scheme's
        own notion of an "event" onto a drag's press and release is one of the
        registered structural deviations: the source papers anchor to a tap, a click or
        a selection on a displayed object, none of which exist in a free drag.
        """
        ev: list[tuple[str, int, int]] = []
        for k, d in enumerate(self.drags):
            ev.append(("press", d.press, k))
            ev.append(("release", d.release, k))
        ev.sort(key=lambda e: e[1])
        return ev

    def degree_to_px(self, value_deg: float) -> float:
        """Realise an angular threshold of the source paper on this display."""
        return float(value_deg) * self.px_per_degree
