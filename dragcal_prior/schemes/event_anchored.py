"""Event-anchored supervision schemes.

Five published schemes whose supervision is built from discrete events -- a tap, a
click, a selection, or the moment a cursor enters a target. Mapping those events onto
the press and release of a drag is the structural deviation shared by every scheme in
this module; it is stated once here rather than repeated per function.

Anchor conventions: ``Pair.frame`` is the frame the pair is anchored to, and ``Pair.t``
the time associated with the supervision sample. Schemes differ in which is which, and that
difference is the temporal-alignment axis of a supervision comparison.
"""

from __future__ import annotations

import numpy as np

from .. import constants
from ..models import PolynomialCalibrationModel
from ..session import Pair, Session
from ..signals import idt_fixations


def gazeswipe(s: Session) -> list[Pair]:
    """GazeSwipe (Cai et al., 2025).

    One pair per drag: the gaze sample taken ``n`` frames before release, paired with
    the cursor at release. The original is designed for thumb swipes on a handheld, so
    three of its elements do not transfer: the reference here is the cursor at release
    rather than a screen object, and the head-pose term is dropped.
    """
    out: list[Pair] = []
    for d in s.drags:
        gi = max(0, d.release - constants.GAZESWIPE_REF_FRAMES_BEFORE_RELEASE)
        out.append(
            Pair(
                gaze_x=float(s.gaze_x[gi]),
                gaze_y=float(s.gaze_y[gi]),
                reference_x=float(s.cursor_x[d.release]),
                reference_y=float(s.cursor_y[d.release]),
                frame=int(d.release),
                t=float(s.t[d.release]),
            )
        )
    return out


def _pace_candidate(s: Session, event_frame: int) -> tuple[float, float, int] | None:
    """The stable gaze segment preceding ``event_frame``, or ``None``.

    A segment qualifies when its gaze speed stays below ``k`` times the median speed of
    the window, lasts at least ``PACE_STABLE_MIN_MS``, and ends within
    ``PACE_END_WITHIN_S`` of the event. The latest qualifying segment wins.
    """
    t = s.t
    lo = int(np.searchsorted(t, t[event_frame] - constants.PACE_WINDOW_S))
    if event_frame - lo < 3:
        return None
    speed = s.speed()[lo:event_frame]
    threshold = float(np.median(speed)) * constants.PACE_SPEED_MEDIAN_K
    ok = speed < threshold if threshold > 0 else np.zeros(len(speed), dtype=bool)
    if not ok.any():
        return None
    idx = np.flatnonzero(ok)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    best: tuple[int, int] | None = None
    for run in runs:
        a, b = lo + int(run[0]), lo + int(run[-1])
        if (t[b] - t[a]) * 1000.0 < constants.PACE_STABLE_MIN_MS:
            continue
        if (t[event_frame] - t[b]) > constants.PACE_END_WITHIN_S:
            continue
        if best is None or b > best[1]:
            best = (a, b)
    if best is None:
        return None
    a, b = best
    return (
        float(np.mean(s.gaze_x[a : b + 1])),
        float(np.mean(s.gaze_y[a : b + 1])),
        b,
    )


def pace(s: Session, model_factory=PolynomialCalibrationModel) -> list[Pair]:
    """PACE (Huang et al., 2016).

    A behavioural check (a stable gaze segment) plus a model-based check that rejects a
    candidate whose prediction under the currently fitted mapping lies more than
    ``PACE_RESIDUAL_DEG`` from the event point.

    ``model_factory`` is injectable so that a caller can supply the mapping the original
    paper used -- a random forest, a hundred trees per axis -- instead of the shared
    polynomial. Twelve head and eye features are reduced to gaze x, y here, because the
    other ten are not part of this data contract.
    """
    out: list[Pair] = []
    threshold = s.degree_to_px(constants.PACE_RESIDUAL_DEG)
    model = model_factory(width=s.screen_width_px, height=s.screen_height_px)
    for _kind, e, _drag in s.events():
        candidate = _pace_candidate(s, e)
        if candidate is None:
            continue
        gx, gy, _b = candidate
        cx, cy = float(s.cursor_x[e]), float(s.cursor_y[e])
        if len(out) >= max(constants.PACE_MODEL_BYPASS_PAIRS, constants.POLY_MIN_PAIRS):
            model.fit(
                np.array([p.gaze_x for p in out]),
                np.array([p.gaze_y for p in out]),
                np.array([p.reference_x for p in out]),
                np.array([p.reference_y for p in out]),
            )
            px, py = model.predict(np.array([gx]), np.array([gy]))
            if float(np.hypot(px[0] - cx, py[0] - cy)) >= threshold:
                continue
        out.append(
            Pair(
                gaze_x=gx,
                gaze_y=gy,
                reference_x=cx,
                reference_y=cy,
                frame=int(e),
                t=float(s.t[e]),
            )
        )
    return out


def online_eye(s: Session) -> list[Pair]:
    """Online-EYE (Hou et al., 2025).

    Walk back from the event to where the cursor entered the target, run I-DT inside
    that span, keep candidate fixations whose centroid lies within
    ``OE_DISTANCE_BOUND_DEG`` of the event point, require the latest candidate to end
    within ``OE_LAST_FIXATION_WITHIN_MS``, and take the nearest.

    Requires ``Session.target_width_px`` and ``Session.target_distance_px``.
    """
    if s.target_width_px is None or s.target_distance_px is None:
        raise ValueError(
            "online_eye needs Session.target_width_px and Session.target_distance_px "
            "(its W/2 entry test); supply them or skip this scheme"
        )
    out: list[Pair] = []
    t = s.t
    dispersion = (
        constants.OE_DISPERSION_RMS_K * constants.OE_DISPERSION_RANGE_FACTOR * s.noise_radius_px()
    )
    max_distance = s.degree_to_px(constants.OE_DISTANCE_BOUND_DEG)
    half_width = s.target_width_px / 2.0
    frame_step = float(np.median(np.diff(t)))

    for kind, e, _drag in s.events():
        if kind == "release":
            inside = s.target_distance_px < half_width
        else:
            # A press has no target yet: the source implementation treats the press
            # point itself as the object centre.
            inside = (
                np.hypot(s.cursor_x - s.cursor_x[e], s.cursor_y - s.cursor_y[e]) < half_width
            )
        lo = e
        while lo - 1 >= 0 and inside[lo - 1]:
            lo -= 1
        if e - lo < 3:
            lo = max(0, e - int(round(0.5 / frame_step)))

        fixations = idt_fixations(
            s.gaze_x,
            s.gaze_y,
            t,
            lo,
            e,
            constants.OE_IDT_MIN_MS / 1000.0,
            dispersion,
        )
        if not fixations:
            continue
        cx, cy = float(s.cursor_x[e]), float(s.cursor_y[e])
        candidates = []
        for a, b in fixations:
            cgx = float(np.mean(s.gaze_x[a : b + 1]))
            cgy = float(np.mean(s.gaze_y[a : b + 1]))
            if np.hypot(cgx - cx, cgy - cy) < max_distance:
                candidates.append((a, b, cgx, cgy))
        if not candidates:
            continue
        if (t[e] - t[candidates[-1][1]]) * 1000.0 > constants.OE_LAST_FIXATION_WITHIN_MS:
            continue
        _a, b, cgx, cgy = min(candidates, key=lambda q: np.hypot(q[2] - cx, q[3] - cy))
        out.append(
            Pair(
                gaze_x=cgx,
                gaze_y=cgy,
                reference_x=cx,
                reference_y=cy,
                frame=int(e),
                t=float(t[b]),
            )
        )
    return out


def zhu2020(s: Session) -> list[Pair]:
    """Zhu et al. (2020).

    Every gaze sample in the 0.33 s before the event is paired with the event point.
    The original fits a Gaussian-process bias field on the last 30 selections; we do not
    bind that cap because a Study-2 unit holds far fewer selections, and eyes are merged
    rather than fitted separately.
    """
    out: list[Pair] = []
    t = s.t
    for _kind, e, _drag in s.events():
        lo = int(np.searchsorted(t, t[e] - constants.ZHU_WINDOW_S))
        cx, cy = float(s.cursor_x[e]), float(s.cursor_y[e])
        for i in range(lo, e):
            out.append(
                Pair(
                    gaze_x=float(s.gaze_x[i]),
                    gaze_y=float(s.gaze_y[i]),
                    reference_x=cx,
                    reference_y=cy,
                    frame=int(e),
                    t=float(t[i]),
                )
            )
    return out


def sidenmark(s: Session) -> list[Pair]:
    """Sidenmark et al. (2019).

    I-DT fixations inside the drag, each centroid paired with the mean cursor over the
    same span. The original paper reports feasibility statistics and implements no
    calibrator, so this scheme has no ``theta`` of its own; the object centre it would
    use is replaced by the cursor.
    """
    out: list[Pair] = []
    t = s.t
    for d in s.drags:
        for a, b in idt_fixations(
            s.gaze_x,
            s.gaze_y,
            t,
            d.press,
            d.release + 1,
            constants.SID_IDT_MIN_MS / 1000.0,
            s.degree_to_px(constants.SID_DISPERSION_DEG),
        ):
            mid = (a + b) // 2
            out.append(
                Pair(
                    gaze_x=float(np.mean(s.gaze_x[a : b + 1])),
                    gaze_y=float(np.mean(s.gaze_y[a : b + 1])),
                    reference_x=float(np.mean(s.cursor_x[a : b + 1])),
                    reference_y=float(np.mean(s.cursor_y[a : b + 1])),
                    frame=int(mid),
                    t=float(t[mid]),
                )
            )
    return out
