# Deviations from the original schemes

Every scheme in this package was re-implemented from its published description and applied
to an apparatus it was not designed for. That always means departing from the original
somewhere. This file records the departures, split into three kinds:

- **Structural** — the original is built around something this data does not contain (a
  displayed stimulus, a tap, a selection on an object). Affects every scheme.
- **Value** — a threshold stated in the original, realised here on a different display.
- **Registered** — the original states no value, or states one that could not be
  transferred. The value used here was fixed *before* the comparison was run.

The point of publishing this is that a reader can tell which numbers came from a paper and
which came from us. A scheme that silently drifts from its original is worse than one whose
drift is on the record.

## Shared structural deviations

**Events are drag presses and releases.** The event-anchored schemes key on a tap, a click
or a selection on a displayed object. None of those exist in an uninstructed drag, so each
scheme's event rule is mapped onto the press and release of the same drags. Where the
original distinguishes event types, both are taken.

**The "stimulus" is the cursor path.** The pursuit-based schemes assume a displayed
stimulus the user has been asked to follow. Here there is no displayed stimulus and no
instruction to follow anything, so the cursor path stands in for it. This is the single
largest deviation in the package and it changes what these schemes measure: gaze–cursor
correlation during a drag is not the same thing as gaze–stimulus correlation during an
elicited pursuit.

**Angular thresholds are realised in pixels.** The source papers state gates in degrees.
They are realised here at 37.7 px/degree on a 1920×1080 display. At the nominal ~33.4
px/degree for that display the realised gate is about 4.5% tighter in visual angle — a
nominal 2° gate is applied at 75.4 px rather than 79.0 px. No comparison reported in the
paper is sensitive to this margin. Constants are marked `[here]` in
`gaze_supervision/constants.py` where this applies.

## Per scheme

### GazeSwipe — Cai et al., 2025

| | |
|---|---|
| Supervision samples | one pair per drag |
| Screening | none |
| Alignment | none |
| Mapping in the original | distance-weighted mean of historical offset vectors |

**Structural.** The original targets thumb swipes on a handheld device. The reference here
is the cursor at release rather than a screen object, and the head-pose term is dropped.

### PACE — Huang et al., 2016

| | |
|---|---|
| Supervision samples | a stable gaze segment within 3 s before the event (≥80 ms, ending within 0.5 s), reduced to one mean vector |
| Screening | behavioural check plus a model-based 2° residual check |
| Alignment | end of the stable segment |
| Mapping in the original | random forest, 100 trees per axis |

**Structural.** Twelve head and eye features are reduced to gaze x and y; the other ten are
not in this data contract.

**Registered.** The model-based check is bypassed until seven pairs have been accepted,
because a fitted model is needed to run it. The original refits every 150 samples; here the
model is refitted at every event. Gaze speed is taken over a three-frame mean, because at
90 Hz roughly half of adjacent frames repeat and the frame-wise median speed is otherwise
zero.

**Note.** `pace()` accepts a `model_factory`, so a caller can supply the random forest of
the original instead of the shared polynomial used here.

### Online-EYE — Hou et al., 2025

| | |
|---|---|
| Supervision samples | look back to where the cursor entered the target; I-DT fixations (70 ms, 1.5× noise radius); the centroid nearest the event |
| Screening | I-DT plus a 10° distance bound |
| Alignment | end of that fixation, within 300 ms of the event |
| Mapping in the original | per-axis affine RLS, forgetting 0.95 → 0.45 |

**Registered.** The 300 ms rule is implemented as a hard gate: the latest candidate fixation
must end within 300 ms of the event. The opposite reading was not run. Forgetting is applied
on the first of ten passes only, to keep the covariance bounded.

**Value.** The original reports its I-DT dispersion in device units. It is expressed here as
1.5 × the tracker's noise radius, converted to a dispersion by
`OE_DISPERSION_RANGE_FACTOR`. Target radius is `W/2` in raw pixels; for a press event, the
distance is measured from the press point.

### Zhu et al. — 2020

| | |
|---|---|
| Supervision samples | every sample in the 0.33 s before the event, each paired with the event point |
| Screening | none |
| Alignment | fixed 0.33 s window |
| Mapping in the original | Gaussian-process bias field (Matérn ν=2.5, length scale 0.9), last 30 selections |

**Structural.** Eyes are merged rather than fitted separately, and the anchor is the cursor
rather than a selected object.

**Registered.** The original keeps the last 30 selections. That cap is not bound here: a
recording unit in this study contains fewer than 30 selections, so it would never trigger.

### Sidenmark et al. — 2019

| | |
|---|---|
| Supervision samples | I-DT fixation centroids within the drag (100 ms, 1°) against the mean cursor over the fixation |
| Screening | I-DT |

**Structural.** The original reports feasibility statistics and implements no calibrator, so
this scheme has no mapping of its own. The object centre it would use is replaced by the
cursor.

### Pursuit Calibration — Pfeuffer et al., 2013

| | |
|---|---|
| Supervision samples | every frame of a 160 ms window with gaze–cursor correlation r > 0.7 on both axes |
| Mapping in the original | RANSAC homography |

**Registered.** The accuracy study reports no threshold; the values are taken from the
paper's application. The registered variant `pursuit_calibration_80ms` uses an 80 ms window
and r > 0.3. The window slides over the whole recording rather than only over an elicited
pursuit episode, for the reason given under shared deviations.

### Smooth-i — Gomez et al., 2018

| | |
|---|---|
| Supervision samples | every frame of a 160 ms window with r > 0.9; a pair is kept only beyond a 1.5° gaze–cursor offset and replaces any earlier pair within 1° |
| Screening | correlation threshold and regional replacement |

**Registered.** Region membership is judged at the reference point, and the first matching
earlier pair is the one replaced.

**Consequence worth knowing.** The 1.5° floor makes this scheme return *nothing* on short
drags where the gaze never gets far enough from the cursor. That is visible in the
synthetic grid and is not an implementation fault.

### Blignaut — 2017

| | |
|---|---|
| Supervision samples | a 100 ms window every 500 ms; its middle 80% of samples, against the mean cursor over the window |
| Screening | cleaning rules; one pruning pass above 1° |
| Mapping in the original | — |

**Structural.** The original runs a concurrent naming task; there is no concurrent task here.

**Registered.** "The middle 80%" is realised as: rank the window's samples by distance to the
window's *median* gaze, keep the closest 80% (at least 3). The original does not say what to
rank by.

**Where the pruning lives.** The 1° residual pruning is a **fitting-stage** step, not a
pair-generation step. `blignaut()` therefore returns unpruned pairs; pass
`prune_residual_deg=1.0` to `gaze_supervision.fitting.fit_mapping` to apply it. The original
does not say how many rounds to run; one is used. If pruning would leave fewer than seven
pairs it is skipped, since the polynomial is not identified below that.

## Not reproduced

**MACGaze — Lei et al., 2025.** Not reproducible on this data contract: it requires face
imagery and a trained appearance model. It is deliberately absent here.
