"""Converts SU2 v8 history.csv force columns into a forces_breakdown.dat
formatted file per anchor case, so SU2BatchCollector.discover() finds and
parses them (v8 did not emit breakdown files for these configs).

Run after (or between) queue runs - idempotent."""
import csv
import glob
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

TEMPLATE = """Total CL:   {cl:.6f}
Total CD:   {cd:.6f}
Total CMz:  {cmz:.6f}
"""


def convert(case_dir: str) -> bool:
    hist = os.path.join(case_dir, "history.csv")
    if not os.path.exists(hist):
        return False
    out = os.path.join(case_dir, "forces_breakdown.dat")
    # always regenerate: an in-progress run may have left stale partial
    # values behind - history.csv is the source of truth
    with open(hist, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False
    last = {k.strip().strip('"').strip(): v
            for k, v in rows[-1].items()}
    try:
        vals = dict(cl=float(last["CL"]), cd=float(last["CD"]),
                    cmz=float(last["CMz"]))
    except (KeyError, ValueError):
        return False
    with open(out, "w") as f:
        f.write(TEMPLATE.format(**vals))
    print("%s -> CL=%.5f CD=%.5f CMz=%+.6f" % (
        os.path.basename(case_dir), vals["cl"], vals["cd"], vals["cmz"]))
    return True


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else ROOT
    n = sum(convert(d) for d in sorted(glob.glob(os.path.join(root, "*"))))
    print(f"converted {n} case(s)")
