# Code Tour — how every piece fits

A guided walk through the repository: what each module does, why it exists,
and where the interesting decisions live. Read top-to-bottom for the full
story, or jump to a layer.

```
                        THE ONE-PARAGRAPH VERSION

Parametric wings are built from CST airfoils + planform laws (geometry/),
evaluated instantly by a verified physics stack (solvers/), bundled into an
engine whose known imperfections are deliberate - it is the cheap prior.
Expensive truth arrives as SU2 runs, passes a physics-bounds quality gate,
and lands in an audited SQLite lake with honest per-column masks
(collector.py -> continual.py). A neural surrogate continually fine-tunes on
that stream - warm-started from the prior, promoted only when holdout improves,
capacity-grown when it plateaus (models/ + continual.py). The multi-fidelity
study (multi_fidelity.py) measures the whole point: accuracy gained per
expensive CFD unit spent. CLI / web / exporters / optimizers are the surfaces.
```

## Layer 0 — geometry (`aerowing/geometry/`)

| File | What it is |
|---|---|
| `cst_3d.py` | `CSTAirfoil3D`: Class-Shape Transformation airfoils — smooth parametric sections fit to NACA 4-digit shapes, differentiable-friendly |
| `wing_3d.py` | `WingSection` + `Wing3D`: loft those sections along a swept/tapered/dihedral/twisted planform; serializes to/from a **37-D design vector** — the language every other layer speaks |
| `benchmarks.py` | ONERA M6, NASA CRM, NACA-swept, supersonic arrow presets. Note the M6 docstring honestly declares its NACA-0010 stand-in airfoil (real M6 root t/c ≈ 9.8% — verified against NASA coordinates) |

Everything downstream consumes `Wing3D`; nothing touches raw geometry math.

## Layer 1 — instant physics (`aerowing/solvers/`)

| File | What it is | Watch out for |
|---|---|---|
| `vlm_3d.py` | Horseshoe VLM: cosine lattice, mirrored-image influence matrix, Trefftz-plane induced drag, PG compressibility frozen at `pg_limit_mach` | The scheme's personality is regression-pinned in `tests/test_vlm_scheme_properties.py`: elliptic e≈1, slope sits below lifting line by ~0.34/AR, reciprocity asymmetry is nx-invariant (legs leave from quarter-chord — mirror symmetry is structurally broken, like every horseshoe VLM) |
| `viscous_3d.py` | Boundary-layer estimate → profile drag from Re/Mach | Semi-empirical; deliberately crude |
| `wave_drag.py` | Korn-Mason drag divergence + Lock 4th-power rise | Known limit: rise onset too soft for thick sections — part of why the old calibration drifted |
| `aero_engine.py` | Couples all three + calibrated transonic lift/drag corrections (`TRANS_LIFT_*`, `TRANS_DRAG_*`) | **The prior, not the product.** Constants are frozen at physically-defensible values (`TRANS_DRAG_C0=0` — the old flat offset was an artifact of fitting mislabeled data). Its residual gap vs CFD is the training signal, by design |
| `su2_3d.py` | Writes production SU2 `.cfg` files | Legacy template options get shimmed for SU2 v8 by the anchor tooling |

## Layer 2 — expensive truth (`cfd_anchors/`, `mesher_3d.py`, `collector.py`)

| Piece | Role |
|---|---|
| `mesher_3d.py` | Single-block structured hexahedral O-grid: y+≈1 wall spacing, TE wake slit, far field at 15 MACs. `mirror_full_span=True` models both semi-spans — required because a free-root half-wing loses ~2/3 of lift vs the wind-tunnel reflection plane (measured: M6 CL 0.084 vs 0.253 experiment) |
| `cfd_anchors/build_anchors.py` | Generates the anchor matrix (M6 alpha sweep, corpus re-runs, grid triplet) with SU2-v8-shimmed configs, 40-D design specs, manifests |
| `cfd_anchors/history_to_forces.py` | SU2 v8 stopped emitting forces breakdowns; this converts history.csv force columns into collector-readable form |
| `collector.py::SU2BatchCollector` | Sweeps run directories, parses forces/residuals, applies `CfdQualityGate` (converged residuals, physical bounds), ingests with **partial masks** — only columns the solver actually delivered |

The gate rejecting garbage loudly is a feature: diverged transonic cases die
on CL/CD/CMz bounds before they can poison anything.

## Layer 3 — memory & learning (`aerowing/continual.py`, `aerowing/models/`)

| Piece | Role |
|---|---|
| `continual.py::AeroDataLake` | SQLite lake: 40-D inputs, 9-D labels, **9-D delivery masks**, paired cheap labels, holdout flags, audit history. `cfd_rows()` exposes only real-CFG-truth rows to validation contexts |
| `CfdQualityGate` | Residual decay + plausibility bounds; reasons recorded forever |
| `GrowableSurrogate` | The novel core: wraps the base network; `add_capacity()` attaches a **zero-initialized residual block** so G'(x) ≡ G(x) exactly at init — function-preserving widening |
| `error_directed_weights` | Formulation C routing: `w = sigmoid(|e_old|/tau) * (1 + relu(|e_old|-|e_new|))` — solved regions get ~half attention, improved regions get reinforced |
| `ContinualTrainer.update()` | Incremental fine-tune from checkpoint with replay buffer (no catastrophic forgetting), optional plateau-triggered growth, holdout promotion gate |
| `models/surrogate_3d.py` | Fourier-feature MLP surrogate (<5 ms inference); physics-regularized loss terms shared with the trainer |
| `models/ensemble_3d.py` | Deep-ensemble UQ bands — rank errors honestly, shrink with labels (re-calibration pending, see Debts) |
| `models/generator_3d.py` | Conditional VAE for generative inverse design |

**Known-fixed bug worth knowing about:** the plateau trigger originally read
zero improvement off any monotone-declining history and grew capacity
mid-improvement. Fixed to window-edge gain; pinned by
`tests/test_growth_policy.py::test_auto_grow_holds_off_while_improving`.

## Layer 4 — proof (`multi_fidelity.py`)

Two truth modes:
- `truth="engine"` — synthetic stand-in; mechanism demo only
- `truth="lake"` — accepted real-CFD rows via `AeroDataLake.cfd_rows()`;
  delivery masks honored end-to-end; undelivered metrics report None

Current measured result (14 trusted anchors, CD RMSE vs real CFD):
VLM-only 0.0345 -> flywheel 0.0150 after 9 labels (-57%), monotone descent,
promotions gated throughout.

## Surfaces

| Path | What |
|---|---|
| `cli/main_cli.py` | Every capability as subcommands (`analyze`, `train`, `export`, `mesher volume`, `continual ...`) |
| `web/server.py` | FastAPI + vendored Three.js studio, localhost-only, refuses AI endpoints without a local checkpoint |
| `optimization/` | Gradient MDO through the surrogate's autograd, NSGA-II Pareto, VAE inverse design |
| `export/` | STL / VTK / SU2 / STEP-CAD writers |
| `configs/*.yaml` | Declarative case definitions (geometry + flight condition) |
| `main.py` | Full-pipeline demonstration script |

## Test map — what guards what

| Test file | Guards |
|---|---|
| `test_vlm_physics.py` | Engine decomposition identity, Trefftz sign, PG continuity |
| `test_vlm_scheme_properties.py` | The discretization's *personality*: elliptic e≈1, slope band, 2π limit, CM arm convergence, bounded nx-invariant reciprocity gap, machine-exact downwash identities, wake insensitivity |
| `test_growth_policy.py` | When capacity growth may/may not fire (plateau vs still-improving vs insufficient CFD mass) |
| `test_continual.py` | Lake semantics, mask integrity (CFD rows REQUIRE explicit masks), gate behavior, function-preserving growth, checkpoint compat |
| `test_multi_fidelity.py` | Prior↔truth label contracts (CDi ~ CL² scaling), lake-truth mode masking |
| `test_collector.py` | Batch sweep idempotency, design-spec parsing, partial masks |
| rest | Geometry, exporters, mesher quality, ensemble UQ, optimization, web API, config |

Run everything: `pytest tests/ -q` (80 passed, 1 xfail-documented as of the
current state).

## Honest debts (also tracked in conversations)

1. O-grid TE/wake drag bias at coarse levels (Richardson trend documented;
   fine-level or twist-aligned wake cut would close it)
2. Twist gradients beyond ~±0.3° total range prevent steady-RANS convergence
   on these O-grids (bisected and documented; anchors use zero twist)
3. Ensemble band shrinkage currently inverted post-engine-fix — xfailed with
   evidence, re-measure against anchors
4. Legacy 30-row corpus quarantined (`legacy_suspect`) — mislabeled physics
   (2D-incompressible / free-root provenance); kept for audit, never trained on

The debts are visible on purpose: the project's thesis is that trustworthy
automation comes from measuring and admitting what you don't know.
