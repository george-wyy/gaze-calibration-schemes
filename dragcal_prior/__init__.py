"""Re-implementations of published gaze-calibration supervision schemes.

Nine scheme entries -- eight published schemes plus one registered variant -- each a pure
function ``Session -> list[Pair]``. Every one is re-implemented from its published
description, and the deviations each makes from its original are documented per scheme,
including which constants are the original's and which were chosen here.

No participant data ships with this package, and none is needed: supply your own
recording as a :class:`~dragcal_prior.session.Session`, or use the synthetic sessions in
``examples/``.

    >>> from dragcal_prior import synthetic_session, run_all
    >>> session = synthetic_session()
    >>> {k: len(v) for k, v in run_all(session).items()}   # doctest: +SKIP
"""

from __future__ import annotations

from .session import Drag, Pair, Session
from .schemes import PRIOR_SCHEMES, PUBLISHED_SCHEMES, run_all, run_scheme

__version__ = "0.1.0"

__all__ = [
    "Drag",
    "Pair",
    "Session",
    "PRIOR_SCHEMES",
    "PUBLISHED_SCHEMES",
    "run_all",
    "run_scheme",
    "synthetic_session",
    "__version__",
]


def synthetic_session(*args, **kwargs) -> Session:
    """Build a synthetic dragging session. See :func:`dragcal_prior.synthetic.session`."""
    from .synthetic import session as _session

    return _session(*args, **kwargs)
