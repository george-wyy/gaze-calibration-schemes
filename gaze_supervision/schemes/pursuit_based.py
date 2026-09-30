"""Pursuit-based supervision schemes.

Three published schemes that supervise from an *elicited pursuit of a displayed
stimulus*: a dot the user is asked to follow. Where that displayed stimulus does not
exist -- an uninstructed drag -- the cursor path stands in for it. That substitution is
the structural deviation shared by every scheme in this module.

Blignaut's residual pruning is deliberately **not** applied here: it is a fitting-stage
step, see :func:`gaze_supervision.fitting.fit_mapping`.
"""

from __future__ import annotations

import numpy as np

from .. import constants
from ..session import Pair, Session
from ..signals import dispersion, pearson


def _pursuit_pairs(s: Session, window_ms: float, correlation_r: float) -> list[Pair]:
    """Sliding-window correlation pairing, the core of Pursuit Calibration and Smooth-i.

    At every frame the gaze and cursor signals over the trailing window must correlate
    above ``correlation_r`` on both axes. The pair is the current frame's gaze against
    the current frame's cursor.
    """
    out: list[Pair] = []
    t = s.t
    window_s = window_ms / 1000.0
    for i in range(s.n):
        lo = int(np.searchsorted(t, t[i] - window_s))
        if i - lo < 3:
            continue
        if pearson(s.gaze_x[lo : i + 1], s.cursor_x[lo : i + 1]) > correlation_r and pearson(
            s.gaze_y[lo : i + 1], s.cursor_y[lo : i + 1]
        ) > correlation_r:
            out.append(
                Pair(
                    gaze_x=float(s.gaze_x[i]),
                    gaze_y=float(s.gaze_y[i]),
                    reference_x=float(s.cursor_x[i]),
                    reference_y=float(s.cursor_y[i]),
                    frame=int(i),
                    t=float(t[i]),
                )
            )
    return out


def pursuit_calibration(s: Session) -> list[Pair]:
    """Pursuit Calibration (Pfeuffer et al., 2013), 160 ms window, r > 0.7.

    The accuracy study reports no threshold; these are the values from that paper's
    application, as registered.
    """
    return _pursuit_pairs(s, constants.PF_WINDOW_MS, constants.PF_CORRELATION_R)


def pursuit_calibration_80ms(s: Session) -> list[Pair]:
    """The registered variant of Pursuit Calibration: 80 ms window, r > 0.3."""
    return _pursuit_pairs(s, constants.PF_WINDOW_MS_VARIANT, constants.PF_CORRELATION_R_VARIANT)


def smooth_i(s: Session) -> list[Pair]:
    """Smooth-i (Gomez et al., 2018).

    Correlated frames (160 ms, r > 0.9) are kept only when the gaze-cursor offset
    exceeds 1.5 degrees, and a new pair replaces an earlier one whose reference point
    lies within 1 degree of it. Replacement is judged at the *reference* point, and the
    first matching earlier pair is the one replaced.
    """
    raw = _pursuit_pairs(s, constants.SM_WINDOW_MS, constants.SM_CORRELATION_R)
    min_offset = s.degree_to_px(constants.SM_MIN_OFFSET_DEG)
    region = s.degree_to_px(constants.SM_REPLACE_REGION_DEG)
    store: list[Pair] = []
    for p in raw:
        if np.hypot(p.gaze_x - p.reference_x, p.gaze_y - p.reference_y) <= min_offset:
            continue
        hit = None
        for index, q in enumerate(store):
            if np.hypot(p.reference_x - q.reference_x, p.reference_y - q.reference_y) <= region:
                hit = index
                break
        if hit is None:
            store.append(p)
        else:
            store[hit] = p
    return store


def blignaut(s: Session) -> list[Pair]:
    """Blignaut (2017).

    A 100 ms window every 500 ms; the 80% of its samples closest to the window median
    are kept, the window is discarded when that subset disperses by more than 5 degrees,
    and the surviving subset's mean gaze is paired with the window's mean cursor.

    The residual pruning the original applies lives in the fitting stage -- call
    :func:`gaze_supervision.fitting.fit_mapping` with ``prune_residual_deg=1.0``.

    The original runs a concurrent naming task and cleans repeatedly; here there is no
    concurrent task and the cleaner runs once.
    """
    out: list[Pair] = []
    t = s.t
    dispersion_threshold = s.degree_to_px(constants.BL_DISPERSION_DEG)
    start = float(t[0])
    end = float(t[-1])
    while start + constants.BL_WINDOW_S <= end:
        a = int(np.searchsorted(t, start))
        b = int(np.searchsorted(t, start + constants.BL_WINDOW_S))
        start += constants.BL_STRIDE_S
        if b - a < 3:
            continue
        gx = s.gaze_x[a:b]
        gy = s.gaze_y[a:b]
        distance = np.hypot(gx - np.median(gx), gy - np.median(gy))
        keep = np.argsort(distance)[: max(3, int(round(constants.BL_KEEP_FRACTION * len(distance))))]
        gx_kept, gy_kept = gx[keep], gy[keep]
        if dispersion(gx_kept, gy_kept) > dispersion_threshold:
            continue
        mid = (a + b) // 2
        out.append(
            Pair(
                gaze_x=float(gx_kept.mean()),
                gaze_y=float(gy_kept.mean()),
                reference_x=float(s.cursor_x[a:b].mean()),
                reference_y=float(s.cursor_y[a:b].mean()),
                frame=int(mid),
                t=float(t[mid]),
            )
        )
    return out
