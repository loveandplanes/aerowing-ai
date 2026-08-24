"""Overnight design-diversity anchor set: four clean transport-like wings,
one cruise point each, through the validated full-span pipeline."""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from aerowing.geometry.cst_3d import CSTAirfoil3D
from aerowing.geometry.wing_3d import Wing3D
from build_anchors import LEVELS, build_case

OUT = os.path.join(REPO, "cfd_anchors", "runs")

# (case_id, AR, taper, sweep, mach, alpha, root_af, tip_af)
SPECS = [
    ("desA_baseline", 9.5, 0.35, 25.0, 0.78, 2.0, "2412", "2410"),
    ("desB_highar",   11.0, 0.30, 30.0, 0.80, 2.5, "2410", "2408"),
    ("desC_lowsweep", 8.5, 0.45, 15.0, 0.84, 2.0, "0012", "0010"),
    ("desD_mid",      10.0, 0.40, 20.0, 0.82, 3.0, "2412", "2412"),
]
SPAN = 34.0
RE = 2.0e7


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    built = []
    for cid, ar, lam, sweep, mach, alpha, af_r, af_t in SPECS:
        if only and only not in cid:
            continue
        wing = Wing3D(
            span=SPAN, aspect_ratio=ar, taper_ratio=lam,
            sweep_le_deg=sweep, dihedral_deg=4.0,
            twist_root_deg=0.0, twist_tip_deg=0.0,
            root_airfoil=CSTAirfoil3D.from_naca4(af_r, order=6),
            tip_airfoil=CSTAirfoil3D.from_naca4(af_t, order=6),
            name=cid.upper())
        man = build_case(cid, wing, alpha, mach, RE, "medium", OUT,
                         extra=dict(group="D_design_diversity"))
        built.append(man)
        print("built %s: %s cells" % (cid, man["mesh_stats"]))
    idx = os.path.join(OUT, "index_designs.json")
    existing = []
    if os.path.exists(idx):
        existing = json.load(open(idx))
    json.dump(existing + built, open(idx, "w"), indent=2)
    print("index:", idx)


if __name__ == "__main__":
    main()
