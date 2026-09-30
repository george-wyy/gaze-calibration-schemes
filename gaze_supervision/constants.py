"""Thresholds used by the reproduced supervision schemes.

Every constant is annotated with its provenance, because the two kinds below are not
the same thing:

* ``[paper]`` -- the value is stated in the cited publication. It was copied, not tuned.
* ``[here]``  -- the cited publication states no value, or states one that cannot be
  transferred as-is (for example an angular gate on a display whose pixel-per-degree
  ratio differs from the original setup). The value below is the one registered before
  the comparison was run.

The distinction matters if you reuse this code: editing a ``[paper]`` value means you
are no longer running that paper's scheme.

Per-scheme deviations from the originals, including structural ones that no constant can
express, are listed in ``docs/DEVIATIONS.md``.
"""

from __future__ import annotations

# --------------------------------------------------------------------- geometry --
# Angular thresholds in the source papers are realised on a 1920x1080 display at 37.7
# px/degree.  [here] -- the original studies report their gates in degrees; the display
# geometry is ours, so the pixel value is derived. At ~33.4 px/degree nominal for this
# display the realised gate is about 4.5% tighter in visual angle than the nominal one
# (a nominal 2 degree gate is applied at 75.4 px rather than 79.0 px).
PX_PER_DEGREE = 37.7
SCREEN_WIDTH_PX = 1920.0
SCREEN_HEIGHT_PX = 1080.0

# Polynomial calibration model used for the model-based checks below, and for the
# unified-theta comparison. Seven coefficients => at least seven pairs.  [here]
POLY_MIN_PAIRS = 7

# ------------------------------------------------------------- GazeSwipe (2025) --
# One pair per drag; gaze taken from the frame before release; mapping is a
# distance-weighted mean of historical offset vectors.  No numeric threshold.
GAZESWIPE_REF_FRAMES_BEFORE_RELEASE = 1  # [here] "the frame before release"

# ------------------------------------------------------------------- PACE (2016) --
PACE_WINDOW_S = 3.0  # [paper] stable gaze segment must lie within 3 s before the event
PACE_STABLE_MIN_MS = 80.0  # [paper] minimum duration of the stable segment
PACE_END_WITHIN_S = 0.5  # [paper] the segment must end within 0.5 s of the event
PACE_RESIDUAL_DEG = 2.0  # [paper] model-based residual check
PACE_SPEED_MEDIAN_K = 1.5  # [paper] stability = gaze speed below 1.5x the window median
# [here] the model-based check is bypassed until this many pairs have been accepted,
# because the check needs a fitted model. The original refits every 150 samples; we
# refit at every event (see docs/DEVIATIONS.md).
PACE_MODEL_BYPASS_PAIRS = 4

# -------------------------------------------------------------- Online-EYE (2025) --
OE_IDT_MIN_MS = 70.0  # [paper] I-DT fixation detection
OE_DISTANCE_BOUND_DEG = 10.0  # [paper] candidate fixation must lie within 10 degrees
OE_LAST_FIXATION_WITHIN_MS = 300.0  # [paper] hard gate on the latest candidate
# [here] I-DT dispersion threshold is expressed as a multiple of the session's gaze
# noise radius, because the original paper reports its dispersion in device units.
OE_DISPERSION_RMS_K = 1.5
# [here] That noise radius is an RMS, whereas I-DT compares a *dispersion* -- the sum of
# the x and y ranges. For n samples of i.i.d. noise the expected range is about 2.97
# sigma at n = 9 (~100 ms at 90 Hz), so the expected dispersion is 2 * 2.97 * sigma,
# which is 4.2 times the RMS radius. Without this factor the threshold is far too tight
# and I-DT finds no fixation at all.
OE_DISPERSION_RANGE_FACTOR = 4.2

# ---------------------------------------------------------------- Zhu et al. (2020) --
ZHU_WINDOW_S = 0.33  # [paper] every sample in the 0.33 s before the event
# [paper] the original keeps the last 30 selections. We do not bind the cap, because a
# Study-2 unit contains fewer than 30 selections; see docs/DEVIATIONS.md.
ZHU_KEEP_LAST_N = 30

# ------------------------------------------------------------- Sidenmark et al. (2019) --
SID_IDT_MIN_MS = 100.0  # [paper] I-DT fixation detection inside the drag
SID_DISPERSION_DEG = 1.0  # [paper] I-DT dispersion threshold

# ------------------------------------------------------ Pursuit Calibration (2013) --
PF_WINDOW_MS = 160.0  # [paper] sliding correlation window
PF_CORRELATION_R = 0.7  # [paper] both-axis correlation threshold
# [paper] the accuracy study reports no threshold; these are the application's variant
PF_WINDOW_MS_VARIANT = 80.0
PF_CORRELATION_R_VARIANT = 0.3

# ----------------------------------------------------------------- Smooth-i (2018) --
SM_WINDOW_MS = 160.0  # [paper] sliding correlation window
SM_CORRELATION_R = 0.9  # [paper] both-axis correlation threshold
SM_MIN_OFFSET_DEG = 1.5  # [paper] a pair is kept only beyond this gaze-cursor offset
SM_REPLACE_REGION_DEG = 1.0  # [paper] an earlier pair within this radius is replaced

# ----------------------------------------------------------------- Blignaut (2017) --
BL_WINDOW_S = 0.1  # [paper] 100 ms window ...
BL_STRIDE_S = 0.5  # [paper] ... taken every 500 ms
BL_KEEP_FRACTION = 0.8  # [paper] middle 80% of the window's samples by distance to median
BL_DISPERSION_DEG = 5.0  # [paper] discard a window above this dispersion
BL_PRUNE_DEG = 1.0  # [paper] one pruning pass above this residual
# [here] the original runs its cleaning pass repeatedly; we run a single round.
BL_PRUNE_ROUNDS = 1
