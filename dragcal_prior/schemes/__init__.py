"""The scheme registry.

``PRIOR_SCHEMES`` holds re-implementations of published supervision schemes. Anything
outside that published set is deliberately absent, so that nothing in this package is
mistaken for a claim about prior work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..session import Pair, Session
from .event_anchored import gazeswipe, online_eye, pace, sidenmark, zhu2020
from .pursuit_based import (
    blignaut,
    pursuit_calibration,
    pursuit_calibration_80ms,
    smooth_i,
)


@dataclass(frozen=True)
class SchemeInfo:
    """One reproduced scheme, with what it needs and where it comes from."""

    key: str
    fn: Callable[[Session], list[Pair]]
    short_cite: str
    anchor: str
    """``event`` = built from discrete press/release events; ``pursuit`` = built from
    correlation between gaze and the (stand-in) stimulus."""
    needs_target_geometry: bool = False
    is_variant: bool = False
    note: str = ""


PRIOR_SCHEMES: dict[str, SchemeInfo] = {
    info.key: info
    for info in (
        SchemeInfo(
            key="gazeswipe",
            fn=gazeswipe,
            short_cite="Cai et al., 2025",
            anchor="event",
            note="One pair per drag.",
        ),
        SchemeInfo(
            key="pace",
            fn=pace,
            short_cite="Huang et al., 2016",
            anchor="event",
            note="Behavioural stability check plus a model-residual gate.",
        ),
        SchemeInfo(
            key="online_eye",
            fn=online_eye,
            short_cite="Hou et al., 2025",
            anchor="event",
            needs_target_geometry=True,
            note="Needs the target width and the cursor-to-target distance.",
        ),
        SchemeInfo(
            key="zhu2020",
            fn=zhu2020,
            short_cite="Zhu et al., 2020",
            anchor="event",
            note="Every sample in the 0.33 s before an event.",
        ),
        SchemeInfo(
            key="sidenmark",
            fn=sidenmark,
            short_cite="Sidenmark et al., 2019",
            anchor="event",
            note="Fixations inside the drag; the original implements no calibrator.",
        ),
        SchemeInfo(
            key="pursuit_calibration",
            fn=pursuit_calibration,
            short_cite="Pfeuffer et al., 2013",
            anchor="pursuit",
            note="160 ms window, r > 0.7.",
        ),
        SchemeInfo(
            key="pursuit_calibration_80ms",
            fn=pursuit_calibration_80ms,
            short_cite="Pfeuffer et al., 2013",
            anchor="pursuit",
            is_variant=True,
            note="Registered variant: 80 ms window, r > 0.3.",
        ),
        SchemeInfo(
            key="smooth_i",
            fn=smooth_i,
            short_cite="Gomez et al., 2018",
            anchor="pursuit",
            note="Correlation plus regional pair replacement.",
        ),
        SchemeInfo(
            key="blignaut",
            fn=blignaut,
            short_cite="Blignaut, 2017",
            anchor="pursuit",
            note="Residual pruning happens at fit time, not here.",
        ),
    )
}

#: The eight published schemes, variants collapsed.
PUBLISHED_SCHEMES = tuple(k for k, v in PRIOR_SCHEMES.items() if not v.is_variant)


def run_scheme(key: str, session: Session) -> list[Pair]:
    """Run one registered scheme by key."""
    if key not in PRIOR_SCHEMES:
        raise KeyError(f"unknown scheme {key!r}; known: {', '.join(PRIOR_SCHEMES)}")
    return PRIOR_SCHEMES[key].fn(session)


def run_all(session: Session, skip_unsatisfied: bool = True) -> dict[str, list[Pair]]:
    """Run every registered scheme.

    With ``skip_unsatisfied`` a scheme whose required session fields are missing is
    skipped rather than raising; the resulting dict simply has no entry for it.
    """
    results: dict[str, list[Pair]] = {}
    for key, info in PRIOR_SCHEMES.items():
        if skip_unsatisfied and info.needs_target_geometry and session.target_width_px is None:
            continue
        results[key] = info.fn(session)
    return results


__all__ = [
    "SchemeInfo",
    "PRIOR_SCHEMES",
    "PUBLISHED_SCHEMES",
    "run_scheme",
    "run_all",
    "gazeswipe",
    "pace",
    "online_eye",
    "zhu2020",
    "sidenmark",
    "pursuit_calibration",
    "pursuit_calibration_80ms",
    "smooth_i",
    "blignaut",
]
