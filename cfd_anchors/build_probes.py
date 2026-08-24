"""Bisection probes: which feature (twist / dihedral / camber) breaks the
full-span mesh convergence on non-M6 wings? Coarse meshes, short runs."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aerowing.geometry.cst_3d import CSTAirfoil3D
from aerowing.geometry.wing_3d import Wing3D
from build_anchors import LEVELS, build_case

OUT = os.path.join(REPO, "cfd_anchors", "runs")
AF_S = CSTAirfoil3D.from_naca4("0012", order=6)
AF_C = CSTAirfoil3D.from_naca4("2412", order=6)

VARIANTS = [
    ("probe_control", dict(twist=(0.0, 0.0), dihedral=0.0, af=AF_S)),
    ("probe_notwist", dict(twist=(0.0, 0.0), dihedral=4.0, af=AF_S)),
    ("probe_nodihedral", dict(twist=(1.5, -1.0), dihedral=0.0, af=AF_S)),
    ("probe_camber", dict(twist=(0.0, 0.0), dihedral=0.0, af=AF_C)),
]

for cid, v in VARIANTS:
    wing = Wing3D(span=34.0, aspect_ratio=9.5, taper_ratio=0.35,
                  sweep_le_deg=25.0, dihedral_deg=v["dihedral"],
                  twist_root_deg=v["twist"][0], twist_tip_deg=v["twist"][1],
                  root_airfoil=v["af"], tip_airfoil=v["af"],
                  name=cid.upper())
    man = build_case(cid, wing, 2.0, 0.78, 2.0e7, "coarse", OUT,
                     extra=dict(group="PROBE"))
    print(cid, man["mesh_stats"])
