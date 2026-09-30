"""A minimal polynomial gaze calibration model.

PACE's model-based check, and a shared-mapping comparison across schemes, both need *some*
mapping from gaze to reference position. The source papers each fit their own mapping
(random forest, homography, Gaussian process, affine RLS); holding one mapping fixed across
schemes keeps the comparison about the supervision and not about the estimator.

The mapping used here is a second-order polynomial with an interaction term, fitted by
ridge regression on display-normalised coordinates:

    phi(x, y) = [1, x, y, x^2, y^2, x*y, x^2*y^2]

Seven coefficients, hence the ``POLY_MIN_PAIRS`` floor of seven pairs. It is **not** part of
any reproduced scheme; it is the common instrument they can be compared through.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import constants


def normalize(x: np.ndarray, y: np.ndarray, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """Map screen pixels to ``[-1, 1]`` on both axes."""
    xn = (np.asarray(x, dtype=float) / width) * 2.0 - 1.0
    yn = (np.asarray(y, dtype=float) / height) * 2.0 - 1.0
    return xn, yn


def design_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The seven-term polynomial basis, shape ``(n, 7)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.column_stack([np.ones_like(x), x, y, x * x, y * y, x * y, x * x * y * y])


@dataclass
class PolynomialCalibrationModel:
    """Ridge-regularised polynomial mapping from gaze to reference position.

    ``ridge`` is a numerical guard, not a tuned hyper-parameter: it keeps the normal
    equations invertible when the pairs are few or nearly collinear. ``[here]``
    """

    ridge: float = 1e-6
    width: float = constants.SCREEN_WIDTH_PX
    height: float = constants.SCREEN_HEIGHT_PX
    coeff_x: np.ndarray | None = field(default=None, repr=False)
    coeff_y: np.ndarray | None = field(default=None, repr=False)

    @property
    def is_fitted(self) -> bool:
        return self.coeff_x is not None and self.coeff_y is not None

    def fit(self, gaze_x, gaze_y, ref_x, ref_y, weights: np.ndarray | None = None) -> "PolynomialCalibrationModel":
        xn, yn = normalize(gaze_x, gaze_y, self.width, self.height)
        a = design_matrix(xn, yn)
        rxn, ryn = normalize(ref_x, ref_y, self.width, self.height)
        if weights is not None:
            w = np.asarray(weights, dtype=float).reshape(-1, 1)
            a = a * np.sqrt(w)
            rxn = rxn * np.sqrt(w.ravel())
            ryn = ryn * np.sqrt(w.ravel())
        gram = a.T @ a + self.ridge * np.eye(a.shape[1])
        self.coeff_x = np.linalg.solve(gram, a.T @ rxn)
        self.coeff_y = np.linalg.solve(gram, a.T @ ryn)
        return self

    def predict(self, gaze_x, gaze_y) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("fit() the model before predicting")
        xn, yn = normalize(gaze_x, gaze_y, self.width, self.height)
        a = design_matrix(xn, yn)
        px = a @ self.coeff_x
        py = a @ self.coeff_y
        return (
            (px + 1.0) * self.width / 2.0,
            (py + 1.0) * self.height / 2.0,
        )

    def residual_px(self, gaze_x, gaze_y, ref_x, ref_y) -> float:
        """Mean Euclidean prediction error in pixels on the supplied pairs."""
        px, py = self.predict(gaze_x, gaze_y)
        return float(np.mean(np.hypot(px - np.asarray(ref_x), py - np.asarray(ref_y))))
