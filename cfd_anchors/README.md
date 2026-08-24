# CFD Anchor Matrix (v4)

The first grid-converged truth set for the continuous-learning flywheel.
Built by `build_anchors.py`; cases live in `runs/<case_id>/`.

## The matrix

| Group | Cases | Purpose |
|---|---|---|
| A `m6_a*__medium` | M6, M=0.8395, Re=11.72e6 (MAC), alpha = 0 / 1.5 / 3.06 / 6 deg | Validates the SU2 setup against the AGARD AR-138 experiment (Case 2308) |
| B `corpus<id>__medium` | 2 wings from the existing lake corpus at their original points | Tests whether the old corpus CD (0.03-0.08) was under-resolved; originals recorded in each manifest |
| C `m6_a3p06__grid_*` | coarse 143k / medium 467k / fine 1.08M cells | Discretization error bars for every other number we quote |

Each case dir contains `mesh_3d.su2` (y+~1 O-grid), `inv_<case>.cfg`
(RANS Spalart-Allmaras), `case.json` (40-D design spec + paired VLM label),
`manifest.json`.

## Workflow

```
python cfd_anchors/build_anchors.py                 # rebuild everything
python cfd_anchors/build_anchors.py --only m6_a3p06 # one case
cd runs/m6_a3p06__medium && SU2_CFD inv_m6_a3p06__medium.cfg
# after all runs:
aerowing continual collect --dir cfd_anchors/runs --source su2_anchor --update
```

The collector quality-gates forces/residuals and ingests with partial masks
(CL/CD/CMz only); source tag `cfd:su2_anchor` is picked up by
`AeroDataLake.cfd_rows()`, so `mf-study --truth lake` consumes anchors
automatically once they land.

## What question each result answers

- **A vs experiment**: is our SU2 setup trustworthy? (CL within ~2-3% of
  Schmitt & Charpin polar at matched alpha)
- **B vs original corpus labels**: were the 30 old CFD rows drag-inflated by
  under-resolution? (compare manifest `original_cd` vs new gated CD)
- **C spread**: how many counts of discretization error ride on every number?
