#!/usr/bin/env python3
"""Run every reproduced scheme over the synthetic Study-2 grid and print pair counts.

    python examples/run_all_schemes.py
    python examples/run_all_schemes.py --csv pairs.csv

Pair counts are the first thing to check when auditing a supervision comparison: a
scheme that supplies two pairs per participant is not being compared under the same
conditions as one that supplies two hundred, whatever the resulting error happens to be.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from gaze_supervision.schemes import PRIOR_SCHEMES, run_all  # noqa: E402
from gaze_supervision.synthetic import grid_sessions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=pathlib.Path, help="also write the table to this file")
    parser.add_argument("--drags", type=int, default=2, help="drags per synthetic session")
    args = parser.parse_args()

    sessions = grid_sessions(n_drags=args.drags)
    keys = list(PRIOR_SCHEMES)

    rows: list[dict[str, object]] = []
    for (distance, width), session in sorted(sessions.items()):
        results = run_all(session)
        row: dict[str, object] = {"distance_px": distance, "width_px": width, "frames": session.n}
        for key in keys:
            row[key] = len(results.get(key, []))
        rows.append(row)

    name_width = max(len(k) for k in keys) + 1
    header = f"{'D':>6} {'W':>5} " + " ".join(f"{k:>{name_width}}" for k in keys)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = " ".join(f"{row[k]:>{name_width}}" for k in keys)
        print(f"{row['distance_px']:>6.0f} {row['width_px']:>5.0f} {cells}")

    never = [k for k in keys if not any(row[k] for row in rows)]
    print()
    if never:
        print(f"WARNING: no output anywhere for: {', '.join(never)}")
        return 1
    print("every scheme produced supervision in at least one grid cell")
    print("note: a zero in a cell is not necessarily a fault -- see docs/DEVIATIONS.md")

    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
