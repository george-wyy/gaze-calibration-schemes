#!/usr/bin/env python3
"""Bring your own recording: build a Session from plain arrays and run the schemes.

    python examples/bring_your_own_data.py

The only requirement is the data contract in ``dragcal_prior.session``: timestamps in
seconds, positions in screen pixels, and the press/release frame indices of each drag.
Nothing here assumes a particular tracker, a participant identifier, or a file format.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dragcal_prior.fitting import fit_mapping  # noqa: E402
from dragcal_prior.session import Drag, Session  # noqa: E402
from dragcal_prior.schemes import run_all  # noqa: E402


def build() -> Session:
    """A hand-made two-second recording with one drag, written out longhand.

    Replace the arrays below with your own. The three things that must be right are that
    ``t`` is strictly increasing, that every array has the same length as ``t``, and
    that each ``Drag`` lies inside the recording.
    """
    fs = 90.0
    n = 180
    t = np.arange(n) / fs

    # A straight diagonal drag from frame 30 to frame 150. It is diagonal rather than
    # horizontal on purpose: the pursuit-based schemes gate on the correlation of gaze
    # and cursor on both axes, and the correlation of a constant signal is undefined, so
    # a drag that moves along one axis only can never pass them.
    cursor_x = np.linspace(400.0, 1200.0, n)
    cursor_y = np.linspace(300.0, 700.0, n)
    cursor_x[:30], cursor_y[:30] = 400.0, 300.0
    cursor_x[150:], cursor_y[150:] = 1200.0, 700.0

    # Gaze follows the cursor with a lead that is large at movement onset and settles to
    # a few milliseconds. A lead that changes faster than this breaks the correlation
    # the pursuit schemes measure -- the signals stop being a shifted copy of each other
    # inside the window -- so the decay here is deliberately gentle.
    lead_frames = 11.0 * np.exp(-np.clip(t - t[30], 0, None) / 0.30) + 0.5
    index = np.clip(np.arange(n) + lead_frames, 0, n - 1)
    gaze_x = np.interp(index, np.arange(n), cursor_x) + np.random.default_rng(0).normal(0, 5, n)
    gaze_y = np.interp(index, np.arange(n), cursor_y) + np.random.default_rng(1).normal(0, 5, n)

    return Session(
        t=t,
        gaze_x=gaze_x,
        gaze_y=gaze_y,
        cursor_x=cursor_x,
        cursor_y=cursor_y,
        drags=[Drag(press=30, release=150)],
        target_width_px=64.0,
        target_distance_px=np.abs(cursor_x - 1200.0),
    ).validate()


def main() -> int:
    session = build()
    print(f"session: {session.n} frames, {len(session.drags)} drag(s), {session.t[-1]:.2f} s")
    print()

    for key, pairs in run_all(session).items():
        if pairs:
            offsets = [np.hypot(p.gaze_x - p.reference_x, p.gaze_y - p.reference_y) for p in pairs]
            print(f"{key:26s} {len(pairs):4d} pairs   median offset {np.median(offsets):6.1f} px")
        else:
            print(f"{key:26s}    0 pairs   (this scheme's gate rejects the whole recording)")

    print()
    pairs = run_all(session)["pursuit_calibration"]
    result = fit_mapping(pairs)
    if result is None:
        print("not enough pairs to fit the seven-coefficient mapping")
        return 0
    print(
        f"fitted the shared mapping on {result.pairs_used} pairs; "
        f"training residual {result.model.residual_px(
            np.array([p.gaze_x for p in pairs]),
            np.array([p.gaze_y for p in pairs]),
            np.array([p.reference_x for p in pairs]),
            np.array([p.reference_y for p in pairs]),
        ):.1f} px"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
