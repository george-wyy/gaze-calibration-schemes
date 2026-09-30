"""Invariants of the reproduced supervision schemes.

These tests are about *contract*, not about reproducing any published number -- those come
from data this package does not ship. What is asserted here is what must hold for the
re-implementations to be trustworthy: that each scheme respects the gate it claims, that
its anchors are the frames it says they are, and that the shared data contract is
enforced rather than assumed.

Runs under pytest, or under ``python tests/run_tests.py`` with no test framework.
"""

from __future__ import annotations

import numpy as np

from _support import approx, raises

from dragcal_prior import constants
from dragcal_prior.fitting import fit_mapping, pairs_to_arrays
from dragcal_prior.session import Drag, Session
from dragcal_prior.schemes import PRIOR_SCHEMES, PUBLISHED_SCHEMES, run_all, run_scheme
from dragcal_prior.synthetic import grid_sessions, session

#: One fixed session, reused by every test below so that they are cheap and comparable.
SESS = session(distance_px=534.0, width_px=64.0, n_drags=3, seed=1)


def _strip_geometry(sess: Session) -> Session:
    return Session(
        t=sess.t,
        gaze_x=sess.gaze_x,
        gaze_y=sess.gaze_y,
        cursor_x=sess.cursor_x,
        cursor_y=sess.cursor_y,
        drags=sess.drags,
    )


# ----------------------------------------------------------------- the contract --
def test_session_rejects_mismatched_shapes() -> None:
    with raises(ValueError, match="same length"):
        Session(
            t=SESS.t,
            gaze_x=SESS.gaze_x[:-1],
            gaze_y=SESS.gaze_y,
            cursor_x=SESS.cursor_x,
            cursor_y=SESS.cursor_y,
            drags=SESS.drags,
        ).validate()


def test_session_rejects_non_monotonic_time() -> None:
    t = SESS.t.copy()
    t[5] = t[4]
    with raises(ValueError, match="strictly increasing"):
        Session(
            t=t,
            gaze_x=SESS.gaze_x,
            gaze_y=SESS.gaze_y,
            cursor_x=SESS.cursor_x,
            cursor_y=SESS.cursor_y,
            drags=SESS.drags,
        ).validate()


def test_drag_rejects_inverted_span() -> None:
    with raises(ValueError, match="invalid drag span"):
        Drag(press=10, release=3)


def test_events_are_time_ordered() -> None:
    events = SESS.events()
    frames = [e[1] for e in events]
    assert frames == sorted(frames)
    assert len(events) == 2 * len(SESS.drags)


# ------------------------------------------------------------------ the registry --
def test_registry_holds_eight_published_schemes() -> None:
    assert len(PUBLISHED_SCHEMES) == 8
    assert set(PUBLISHED_SCHEMES) == {
        "gazeswipe",
        "pace",
        "online_eye",
        "zhu2020",
        "sidenmark",
        "pursuit_calibration",
        "smooth_i",
        "blignaut",
    }


def test_run_all_covers_every_registered_scheme() -> None:
    assert set(run_all(SESS)) == set(PRIOR_SCHEMES)


def test_run_scheme_rejects_unknown_key() -> None:
    with raises(KeyError, match="unknown scheme"):
        run_scheme("not_a_scheme", SESS)


def test_schemes_are_deterministic() -> None:
    first = {(k, p.frame) for k, v in run_all(SESS).items() for p in v}
    second = {(k, p.frame) for k, v in run_all(SESS).items() for p in v}
    assert first == second


# ------------------------------------------------------------------- per scheme --
def test_gazeswipe_takes_one_pair_per_drag_from_the_frame_before_release() -> None:
    pairs = run_scheme("gazeswipe", SESS)
    assert len(pairs) == len(SESS.drags)
    for pair, drag in zip(pairs, SESS.drags):
        assert pair.frame == drag.release
        assert pair.gaze_x == approx(SESS.gaze_x[drag.release - 1])
        assert pair.reference_x == approx(SESS.cursor_x[drag.release])


def test_zhu2020_pairs_only_samples_before_an_event() -> None:
    pairs = run_scheme("zhu2020", SESS)
    assert pairs, "the fixture should supply supervision to every scheme"
    event_frames = {frame for _kind, frame, _drag in SESS.events()}
    for pair in pairs:
        assert pair.frame in event_frames
        # t is the sample's own time, so it must precede the event it is paired with
        assert pair.t < SESS.t[pair.frame]


def test_online_eye_requires_target_geometry() -> None:
    with raises(ValueError, match="target_width_px"):
        run_scheme("online_eye", _strip_geometry(SESS))


def test_online_eye_anchors_on_events_and_bounds_its_fixations() -> None:
    pairs = run_scheme("online_eye", SESS)
    assert pairs
    event_frames = {frame for _kind, frame, _drag in SESS.events()}
    max_distance = SESS.degree_to_px(constants.OE_DISTANCE_BOUND_DEG)
    for pair in pairs:
        assert pair.frame in event_frames
        offset = np.hypot(pair.gaze_x - pair.reference_x, pair.gaze_y - pair.reference_y)
        assert offset < max_distance


def test_pace_pairs_are_bounded_by_the_event_window() -> None:
    pairs = run_scheme("pace", SESS)
    assert pairs
    for pair in pairs:
        assert SESS.t[0] <= pair.t <= SESS.t[-1]


def test_sidenmark_pairs_are_inside_drags() -> None:
    pairs = run_scheme("sidenmark", SESS)
    assert pairs
    for pair in pairs:
        assert any(d.press <= pair.frame <= d.release for d in SESS.drags)


def _assert_pursuit_gate(key: str, window_ms: float, r_threshold: float) -> None:
    pairs = run_scheme(key, SESS)
    assert pairs
    for pair in pairs:
        lo = int(np.searchsorted(SESS.t, SESS.t[pair.frame] - window_ms / 1000.0))
        assert pair.frame - lo >= 3
        for axis_gaze, axis_cursor in ((SESS.gaze_x, SESS.cursor_x), (SESS.gaze_y, SESS.cursor_y)):
            a = axis_gaze[lo : pair.frame + 1]
            b = axis_cursor[lo : pair.frame + 1]
            assert float(np.corrcoef(a, b)[0, 1]) > r_threshold


def test_pursuit_pairs_survive_their_own_correlation_gate() -> None:
    assert PRIOR_SCHEMES["pursuit_calibration"].is_variant is False
    _assert_pursuit_gate("pursuit_calibration", constants.PF_WINDOW_MS, constants.PF_CORRELATION_R)


def test_pursuit_variant_pairs_survive_their_own_correlation_gate() -> None:
    assert PRIOR_SCHEMES["pursuit_calibration_80ms"].is_variant is True
    _assert_pursuit_gate(
        "pursuit_calibration_80ms",
        constants.PF_WINDOW_MS_VARIANT,
        constants.PF_CORRELATION_R_VARIANT,
    )


def test_smooth_i_keeps_only_large_offsets_and_deduplicates_regions() -> None:
    pairs = run_scheme("smooth_i", SESS)
    assert pairs, "the fixture should exercise this scheme on the default cell"
    min_offset = SESS.degree_to_px(constants.SM_MIN_OFFSET_DEG)
    region = SESS.degree_to_px(constants.SM_REPLACE_REGION_DEG)
    for pair in pairs:
        assert np.hypot(pair.gaze_x - pair.reference_x, pair.gaze_y - pair.reference_y) > min_offset
    for i, a in enumerate(pairs):
        for b in pairs[i + 1 :]:
            distance = np.hypot(a.reference_x - b.reference_x, a.reference_y - b.reference_y)
            assert distance > region


def test_blignaut_pairs_lie_inside_the_session() -> None:
    pairs = run_scheme("blignaut", SESS)
    assert pairs
    for pair in pairs:
        assert SESS.t[0] <= pair.t <= SESS.t[-1]


def test_every_scheme_produces_pairs_somewhere_on_the_grid() -> None:
    """A scheme that never fires anywhere would be silently dead code."""
    produced: set[str] = set()
    for sess in grid_sessions(n_drags=2).values():
        for key, pairs in run_all(sess).items():
            if pairs:
                produced.add(key)
    assert produced == set(PRIOR_SCHEMES), f"never fired: {set(PRIOR_SCHEMES) - produced}"


# -------------------------------------------------------------------- fitting --
def test_fit_mapping_needs_seven_pairs() -> None:
    pairs = run_scheme("gazeswipe", SESS)
    assert len(pairs) == 3
    assert fit_mapping(pairs) is None


def test_fit_mapping_recovers_a_fitted_model() -> None:
    pairs = run_scheme("pursuit_calibration", SESS)
    result = fit_mapping(pairs)
    assert result is not None
    assert result.pairs_used == len(pairs)
    assert result.model.is_fitted


def test_fit_mapping_prune_reduces_pairs_and_keeps_the_model() -> None:
    pairs = run_scheme("pursuit_calibration", SESS)
    unpruned = fit_mapping(pairs)
    pruned = fit_mapping(pairs, prune_residual_deg=constants.BL_PRUNE_DEG)
    assert unpruned is not None and pruned is not None
    assert pruned.pairs_used <= unpruned.pairs_used
    assert pruned.pairs_dropped == len(pairs) - pruned.pairs_used
    assert pruned.model.is_fitted


def test_pairs_to_arrays_round_trips() -> None:
    pairs = run_scheme("zhu2020", SESS)
    gx, gy, rx, ry = pairs_to_arrays(pairs)
    assert len(gx) == len(gy) == len(rx) == len(ry) == len(pairs)
    assert gx[0] == approx(pairs[0].gaze_x)
    assert ry[-1] == approx(pairs[-1].reference_y)
