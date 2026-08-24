"""Builds the CFD anchor matrix for v4: the first grid-converged truth set.

Groups
  A  m6_alpha     ONERA M6, M=0.8395, Re=11.72e6 (MAC), alpha in {0,1.5,3.06,6} deg
                  -> validates the SU2 setup against AGARD AR-138 experiment
  B  corpus_rerun two wings from the existing lake corpus, re-meshed at
                  anchor quality at their original conditions
                  -> tests whether the old corpus was under-resolved
  C  grid_triplet M6 alpha=3.06 at three mesh refinements
                  -> discretization error bars for everything else

Each case directory receives: mesh_3d.su2 (y+~1 O-grid), inv_<case>.cfg,
case.json (40-D design spec + paired VLM label), manifest.json.
After SU2 finishes:  aerowing continual collect --dir <out> --source su2_anchor
(the collector quality-gates forces and ingests with partial masks).
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from aerowing.geometry.benchmarks import get_onera_m6_wing
from aerowing.geometry.wing_3d import Wing3D
from aerowing.mesher_3d import VolumeMesher3D
from aerowing.multi_fidelity import evaluate_low_fi
from aerowing.solvers.su2_3d import SU2Driver3D

M6_MACH = 0.8395
M6_RE_MAC = 11.72e6

# mesh level -> mesher kwargs + SU2 iteration budget
LEVELS = {
    "coarse": dict(n_loop=48, n_stations=24, y_plus=1.0, iters=2000),
    "medium": dict(n_loop=96, n_stations=48, y_plus=1.0, iters=5000),
    "fine": dict(n_loop=160, n_stations=72, y_plus=1.0, iters=10000),
}


def _mesh_stats(mesh):
    out = {}
    for attr in ("n_nodes", "n_cells", "nodes", "cells"):
        v = getattr(mesh, attr, None)
        if v is not None:
            out[attr] = int(v) if np.isscalar(v) else len(v)
    return out


def build_case(case_id, wing, alpha, mach, reynolds, level, out_root,
               extra=None, skip_mesh=False):
    """Meshes one case, writes cfg + design spec + manifest."""
    lvl = LEVELS[level]
    d = os.path.join(out_root, case_id)
    os.makedirs(d, exist_ok=True)

    mesh = None
    if not skip_mesh:
        mesher = VolumeMesher3D(wing, n_loop=lvl["n_loop"],
                                n_stations=lvl["n_stations"],
                                y_plus=lvl["y_plus"], reynolds=reynolds,
                                mirror_full_span=True)
        mesh = mesher.build()
        mesh.export_su2(os.path.join(d, "mesh_3d.su2"))

    driver = SU2Driver3D(wing)
    cfg_path = os.path.join(d, f"inv_{case_id}.cfg")
    driver.generate_config(mach=mach, alpha_deg=alpha, reynolds=reynolds,
                           solver_type="RANS", turbulence_model="SA",
                           num_iterations=lvl["iters"],
                           output_filepath=cfg_path)
    # the O-grid mesher emits only `wing` and `farfield` markers (the root
    # plug is closed off to the far field), so the template's symmetry
    # marker would be a fatal unknown-marker error in SU2.
    with open(cfg_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.strip().startswith("MARKER_SYM")]
    # SU2 v8 renames/shorthands relative to the legacy driver template
    SHIM = {
        "KIND_TURB_MODEL= SPALART_ALLMARAS": "KIND_TURB_MODEL= SA",
        "REF_SEMI_SPAN": "SEMI_SPAN",
    }
    out = []
    for ln in lines:
        for old, new in SHIM.items():
            if ln.strip().startswith(old.split("=")[0].strip()) and old in ln:
                ln = ln.replace(old, new)
        out.append(ln)
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.writelines(out)

    x40 = np.concatenate([
        np.asarray(wing.to_parameter_vector(), dtype=float),
        [float(alpha), float(mach), float(np.log10(max(reynolds, 1e4)))]])
    y_vlm = evaluate_low_fi(wing, float(alpha), float(mach),
                            float(reynolds)).tolist()
    with open(os.path.join(d, "case.json"), "w", encoding="utf-8") as f:
        json.dump({"x": x40.tolist(), "y_vlm": y_vlm}, f)

    manifest = dict(case_id=case_id, level=level,
                    wing=wing.name, alpha_deg=float(alpha), mach=float(mach),
                    reynolds=float(reynolds), re_length="MAC",
                    solver="RANS-SA", mesh_stats=_mesh_stats(mesh) if mesh
                    else "skipped",
                    mesh_convention="full_span_mirror" if mesh else "n/a",
                    files=["inv_%s.cfg" % case_id]
                    + ([] if skip_mesh else ["mesh_3d.su2"])
                    + ["case.json"],
                    run_cmd=f"SU2_CFD inv_{case_id}.cfg",
                    **(extra or {}))
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(REPO, "cfd_anchors", "runs"))
    ap.add_argument("--lake", default=os.path.join(REPO, "data_lake",
                                                   "aero.sqlite"))
    ap.add_argument("--levels", default="medium",
                    help="comma list among coarse,medium,fine (group A/B)")
    ap.add_argument("--corpus-ids", default=None,
                    help="two lake row ids for group B; default = 5th and "
                         "15th accepted CFD rows")
    ap.add_argument("--only", default=None,
                    help="build a single case id substring (smoke tests)")
    ap.add_argument("--skip-mesh", action="store_true",
                    help="write cfg/spec/manifest only")
    args = ap.parse_args()
    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    for lv in levels:
        if lv not in LEVELS:
            raise SystemExit(f"unknown level '{lv}'")

    cases = []

    # A: M6 alpha sweep (anchor-quality medium mesh per point)
    m6 = get_onera_m6_wing()
    for a in (0.0, 1.5, 3.06, 6.0):
        for lv in levels:
            cases.append((f"m6_a{str(a).replace('.', 'p')}__{lv}", m6, a,
                          M6_MACH, M6_RE_MAC, lv,
                          dict(group="A_m6_alpha")))

    # B: corpus re-runs at their original conditions
    from aerowing.continual import AeroDataLake
    lake = AeroDataLake(args.lake)
    try:
        x_all, y_all, m_all, ids = lake.cfd_rows()
    finally:
        lake.close()
    if len(ids) >= 15:
        pick = ([int(v) for v in args.corpus_ids.split(",")]
                if args.corpus_ids else [ids[4], ids[14]])
        id_pos = {rid: i for i, rid in enumerate(ids)}
        for rid in pick:
            xv = x_all[id_pos[rid]]
            wing = Wing3D.from_parameter_vector(np.asarray(xv[:37]),
                                                name=f"corpus_rerun_{rid}")
            for lv in levels:
                cases.append((
                    f"corpus{rid}__{lv}", wing, float(xv[37]), float(xv[38]),
                    float(10 ** xv[39]), lv,
                    dict(group="B_corpus_rerun", lake_row_id=int(rid),
                         original_cd=float(y_all[id_pos[rid]][1]),
                         original_cl=float(y_all[id_pos[rid]][0]))))
    else:
        print(f"[anchors] lake has {len(ids)} CFD rows; skipping group B")

    # C: grid triplet on M6 alpha=3.06
    for lv in ("coarse", "medium", "fine"):
        cases.append((f"m6_a3p06__grid_{lv}", get_onera_m6_wing(), 3.06,
                      M6_MACH, M6_RE_MAC, lv, dict(group="C_grid_triplet")))

    if args.only:
        cases = [c for c in cases if args.only in c[0]]
        if not cases:
            raise SystemExit(f"--only '{args.only}' matched nothing")

    print(f"[anchors] building {len(cases)} cases under {args.out}")
    built = []
    for cid, wing, a, m, re, lv, extra in cases:
        man = build_case(cid, wing, a, m, re, lv, args.out, extra=extra,
                         skip_mesh=args.skip_mesh)
        built.append(man)
        print(f"  {cid:<28} M={m:.4f} a={a:5.2f} Re={re:.3e} "
              f"mesh={man['mesh_stats']}")

    index_path = os.path.join(args.out, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(built, f, indent=2)
    print(f"\n[anchors] index written: {index_path}")
    print("[anchors] next steps:")
    print(f"  1. cd into each case dir and run:  SU2_CFD inv_<case>.cfg")
    print(f"  2. collect through the quality gate:")
    print(f"     aerowing continual collect --dir {args.out} "
          f"--source su2_anchor --update")


if __name__ == "__main__":
    main()
