"""Fitting stage: turning supervision pairs into a gaze-to-reference mapping.

The schemes in this package decide *which samples become supervision*. What is done
with those samples -- which mapping is fitted -- is a separate axis. These helpers hold the
mapping fixed across schemes (a second-order polynomial, see :mod:`.models`) so that a
comparison is about the supervision and not about the estimator.

One scheme does not fit anywhere else: Blignaut (2017) drops pairs whose residual under
a first fit exceeds one degree, then refits. That is a fitting-stage step, not a
pair-generation step, so it lives here rather than inside the scheme function. The
library keeps both stages separate so that a caller can see exactly how many pairs the
first fit used and how many survived.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants
from .models import PolynomialCalibrationModel
from .session import Pair


@dataclass
class FitResult:
    """A fitted mapping plus the provenance of the pairs it was fitted on."""

    model: PolynomialCalibrationModel
    pairs_used: int
    pairs_supplied: int
    pruned: bool = False

    @property
    def pairs_dropped(self) -> int:
        return self.pairs_supplied - self.pairs_used


def pairs_to_arrays(pairs: list[Pair]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack pairs into ``(gaze_x, gaze_y, reference_x, reference_y)`` arrays."""
    gx = np.array([p.gaze_x for p in pairs], dtype=float)
    gy = np.array([p.gaze_y for p in pairs], dtype=float)
    rx = np.array([p.reference_x for p in pairs], dtype=float)
    ry = np.array([p.reference_y for p in pairs], dtype=float)
    return gx, gy, rx, ry


def fit_mapping(
    pairs: list[Pair],
    model: PolynomialCalibrationModel | None = None,
    prune_residual_deg: float | None = None,
    min_pairs: int = constants.POLY_MIN_PAIRS,
    width: float = constants.SCREEN_WIDTH_PX,
    height: float = constants.SCREEN_HEIGHT_PX,
) -> FitResult | None:
    """Fit the shared mapping to ``pairs``.

    Parameters
    ----------
    prune_residual_deg:
        When given, run one round of residual pruning: fit, drop pairs whose prediction
        error exceeds this angle, then refit. This is the registered realisation of the
        Blignaut (2017) cleaning step. The prune is skipped when it would leave fewer
        than ``min_pairs`` pairs, matching the original pipeline.
    min_pairs:
        Fewer pairs than this returns ``None``: the polynomial has seven coefficients
        and is not identified below that.

    Returns ``None`` when the fit is not possible.
    """
    if len(pairs) < min_pairs:
        return None
    gx, gy, rx, ry = pairs_to_arrays(pairs)
    if model is None:
        model = PolynomialCalibrationModel(width=width, height=height)
    model.fit(gx, gy, rx, ry)
    result = FitResult(model=model, pairs_used=len(pairs), pairs_supplied=len(pairs))

    if prune_residual_deg is not None:
        predicted_x, predicted_y = model.predict(gx, gy)
        keep = np.hypot(predicted_x - rx, predicted_y - ry) <= (
            prune_residual_deg * constants.PX_PER_DEGREE
        )
        kept = int(keep.sum())
        if min_pairs <= kept < len(keep):
            model.fit(gx[keep], gy[keep], rx[keep], ry[keep])
            result.pairs_used = kept
            result.pruned = True
    return result
