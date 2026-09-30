# gaze-calibration-schemes

Re-implementations of eight published gaze-calibration supervision schemes, plus one
registered variant.

Each scheme is a pure function from a recorded session to the supervision pairs it would
extract. The re-implementations follow the published descriptions; the deviations each one
makes from its original are documented per scheme in
[docs/DEVIATIONS.md](docs/DEVIATIONS.md).

| key | scheme | source | anchors on |
|---|---|---|---|
| `gazeswipe` | GazeSwipe | Cai et al., 2025 | discrete events |
| `pace` | PACE | Huang et al., 2016 | discrete events |
| `online_eye` | Online-EYE | Hou et al., 2025 | discrete events |
| `zhu2020` | Zhu et al. | Zhu et al., 2020 | discrete events |
| `sidenmark` | Sidenmark et al. | Sidenmark et al., 2019 | discrete events |
| `pursuit_calibration` | Pursuit Calibration | Pfeuffer et al., 2013 | gaze–cursor correlation |
| `pursuit_calibration_80ms` | *variant of the above* | Pfeuffer et al., 2013 | gaze–cursor correlation |
| `smooth_i` | Smooth-i | Gomez et al., 2018 | gaze–cursor correlation |
| `blignaut` | Blignaut | Blignaut, 2017 | gaze–cursor correlation |

Full citations are in [docs/REFERENCES.md](docs/REFERENCES.md). MACGaze (Lei et al., 2025)
is not reproduced here: it requires face imagery and a trained appearance model, which this
data contract does not cover.

No participant data is included, and none is needed: the repository ships a synthetic
Fitts-grid fixture and a test suite that runs without a test framework.

## Usage

Only Python and numpy are required.

```bash
git clone https://github.com/george-wyy/gaze-calibration-schemes
cd gaze-calibration-schemes
pip install numpy          # the only dependency

python examples/run_all_schemes.py        # pair counts over a synthetic grid
python examples/bring_your_own_data.py    # build a session from your own arrays
python tests/run_tests.py                 # 23 checks, no test framework needed
```

```python
from gaze_supervision import synthetic_session, run_all

session = synthetic_session(distance_px=534, width_px=64)
for scheme, pairs in run_all(session).items():
    print(f"{scheme:26s} {len(pairs):5d} pairs")
```

## The data contract

A scheme consumes a `Session` and returns a list of `Pair`. Nothing else.

```python
Session(
    t,                    # frame timestamps, seconds, strictly increasing
    gaze_x, gaze_y,       # gaze position, screen pixels
    cursor_x, cursor_y,   # cursor position, screen pixels
    drags=[Drag(press=30, release=150), ...],
    px_per_degree=37.7,
    target_width_px=64.0,          # Online-EYE only
    target_distance_px=...,        # Online-EYE only
)
```

`Pair(gaze_x, gaze_y, reference_x, reference_y, frame, t)` is one supervision sample: what
the gaze was, and what the reference position was. Which frame and which timestamp a pair is
anchored to differs between schemes, and it is visible in the returned `frame` and `t` fields
rather than hidden inside the code.

See [examples/bring_your_own_data.py](examples/bring_your_own_data.py) for a worked example.

## Documentation

| | |
|---|---|
| [docs/DEVIATIONS.md](docs/DEVIATIONS.md) | Where each re-implementation departs from its original, and which constants are the original's versus ours |
| [docs/REFERENCES.md](docs/REFERENCES.md) | Full citations for the schemes |
| [gaze_supervision/constants.py](gaze_supervision/constants.py) | Every threshold, annotated `[paper]` or `[here]` |

## Citation

Please cite the original authors of the schemes — see
[docs/REFERENCES.md](docs/REFERENCES.md) — as well as this repository.

## Questions

For technical questions, please open a GitHub issue.

## License

MIT — see [LICENSE](LICENSE). The re-implementations are original code; the schemes they
implement belong to their cited authors.
