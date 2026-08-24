# AeroWing AI Pro v4

Enterprise-Grade 3D Aerodynamic AI Design, CFD Surrogate & MDO Optimization Suite.

## Evidence (validated end-to-end)

**Physics core under adversarial verification.** The 3D VLM survived a
structured audit: analytic anchors (elliptic planform span efficiency = 1.002,
lift slope within the horseshoe band, 2*pi/rad thin-airfoil limit recovered at
AR=200), machine-exact Trefftz identities, and regression-pinned scheme
behavior (`tests/test_vlm_scheme_properties.py`).

**SU2 anchor setup validated against experiment.** Grid-converged RANS on
ONERA M6 reproduces the AGARD AR-138 polar: CL = 0.134 / 0.271 / 0.493 at
alpha = 1.5 / 3.06 / 6 deg (M = 0.8395, Re = 11.7e6), plus a coarse-to-medium
refinement trend for discretization error bars.

**The flywheel learns from real CFD.** Money slide on quality-gated SU2
anchors (9 candidate designs / 5 holdout, CD RMSE vs truth):

```
  CFD units | flywheel | vs frozen preliminary tool
          0 |  0.03167 |   -8%
          2 |  0.02927 |  -15%
          4 |  0.02749 |  -20%
          9 |  0.01496 |  -57%   (and every new run shrinks it further)
```

Every link is instrumented: diverged runs are rejected by physics-bounds
gates, partial solver deliveries train only delivered columns, promotion is
holdout-gated, and capacity growth preserves the learned function exactly.

## Features

* 3D Vortex Lattice Method (VLM) with Prandtl-Glauert compressibility correction
* CST (Class-Shape Transformation) 3D wing parameterization with benchmark geometries
  (ONERA M6, NASA CRM)
* Viscous corrections, wave drag estimation, and multi-fidelity aero engine
* ML surrogate models, generative 3D-CVAE inverse design, and NSGA-II Pareto MDO
* Continuous learning engine: every CFD run you do anyway becomes a training
  label (quality-gated, exploration-balanced, holdout-gated promotion,
  incremental fine-tune with replay, automatic capacity growth)
* Multi-fidelity study harness: the error-vs-CFD-cost curve proving the VLM-warm
  flywheel beats a cold surrogate at the same label budget
* STEP / STL / SU2 / VTK exporters and a FastAPI web UI

## Installation

```bash
python -m pip install -e ".[dev]"
```

Requires Python >= 3.10 and NumPy >= 2.0.

## Usage

```bash
aerowing info
aerowing analyze --case onera_m6
aerowing analyze --config onera_m6        # load geometry/flight condition from configs/*.yaml
aerowing train
aerowing inverse-design --target-cl 0.55 --target-mach 0.82
aerowing serve                            # private localhost-only web UI
aerowing continual status                 # data lake statistics
aerowing continual ingest --forces run.log --design-json design.json   # quality-gated label
aerowing continual collect --dir cfd_runs/ --update     # batch-ingest all runs, then improve
aerowing continual update --auto-grow     # fine-tune + holdout-gated promotion
aerowing continual mf-study --designs 256 --holdout 64 --budgets "0,32,128,256" \
    --out study.json                       # error-vs-CFD-cost money slide
python main.py            # full validation demonstration
```

## Continuous Learning (the tool keeps improving)

Every CFD run your team performs anyway can teach the surrogate. The loop:

1. **CFD run -> quality gate** (`aerowing/continual.py`): only converged runs
   (residuals decreasing, final below 1e-5) with physically plausible forces
   (|CL| bounds, 0 < CD) enter the data lake. Divergent or garbage runs are
   rejected with recorded reasons and never trained on. Batch mode
   (`aerowing continual collect --dir cfd_runs/`) sweeps whole output
   directories automatically and is idempotent — safe to run on a schedule.
2. **Exploration tax**: a randomized design is injected every N proposals so
   the dataset keeps covering the design space instead of collapsing onto the
   current optimum.
3. **Incremental fine-tune**: the surrogate is refined from the last checkpoint
   (small LR) with a replay buffer of older samples — no catastrophic
   forgetting, no retrain-from-scratch.
4. **Holdout promotion gate**: a fixed holdout set (never trained on) decides
   promotion; a checkpoint that regresses on it is refused and the previous
   one is kept. All of it audited in the SQLite data lake.
5. **Capacity growth** (airfoil_ai-inspired error-directed routing): when the
   corpus has grown and holdout improvement plateaus, the network grows with a
   zero-initialized residual block (old function preserved exactly at init)
   and fine-tunes with error-directed per-sample gradient weights —
   solved regions stay untouched while needing regions learn.

The web server automatically loads the promoted checkpoint
(`checkpoints/aerowing_models.pt`).

## Multi-Fidelity Study (the money slide)

`aerowing continual mf-study` quantifies what learning is worth in CFD cost.
It samples a design space, spans it with the cheap VLM solver, then spends CFD
"units" (high-fidelity `AeroEngine3D` labels) through the continuous-learning
loop. Output: CD RMSE on unseen holdout designs versus CFD units spent, for
two learners — a **flywheel** (warm-started from the VLM-pretrained surrogate,
which is what this project does) and a **cold** surrogate trained only on the
expensive labels. Domain is a cruise flight band (alpha 1..4.5 deg, M 0.76..0.85,
Re ~1.6e7..3.2e7) where a 37-D design corpus can be resolved.

Example output (256 candidates, 64 holdout, seed 1337):

```
  CFD units | promoted | flywheel |   cold   | vs VLM-only
          0 |      no  | 0.01462  |    n/a   |    69.9%
         32 |      yes | 0.01278  | 0.01745  |    48.5%
        128 |      yes | 0.01266  | 0.01660  |    47.1%
        256 |      yes | 0.00867  | 0.01946  |     0.8%
```

The VLM-warm flywheel dominates the cold learner at every budget and converges
to the VLM-only reference (~0.0086) — i.e., a few hundred CFD runs turn a
global 5 ms surrogate into a design-screening tool with turbofan-level drag
bedrock error, while the expensive labels alone go nowhere without the cheap
physics prior. The same JSON (`--out`) feeds the per-column tables and the
ASCII curve.

**Truth modes.** `--truth engine` (default) exercises the mechanism against
the synthetic `AeroEngine3D` stand-in — useful for comparing learner recipes,
but it is *not* a claim about real-CFD accuracy, since the stand-in shares its
physics with the cheap prior. `--truth lake --lake-path data_lake/aero.sqlite`
runs the same sweep with accepted real CFD rows as ground truth: each row's
delivered-columns mask is honored end to end (training, holdout scoring and
reported RMSE), and undelivered outputs report `n/a` instead of fabricated
errors. This is the mode that demonstrates the thesis on real data; the
engine mode exists so the harness stays runnable before anchors accrue.

## Volume Mesh Generator (CFD fuel)

`aerowing mesher volume` builds a structured hexahedral O-grid volume mesh
around any parametric wing and writes it as a SU2 mesh — the input side of
the continuous-learning loop (CFD runs need meshes; the mesh farm makes the
loop operable on demand).

- **y+ -resolved wall spacing**: the first cell height comes from a flat-plate
  skin-friction law for the target y+ (default 1); the layer count is solved
  automatically from the growth ratio and the far-field radius, so the wall
  cell lands at or below the target y+.
- **O-grid topology with TE slit**: closed ring around each station, slit split
  at the trailing edge, blunt-TE clamping so the wedge cells never fold, and
  a coarse/verification `--coarse` preset for quick sanity runs.
- **Boundary conditions**: `wing` (viscous wall) and `farfield` markers,
  root plug extrusion and tip cone that shrinks to a point on the far field.

```
aerowing mesher volume --wing onera_m6 --out mesh_onera_m6.su2
aerowing mesher volume --wing onera_m6 --coarse                  # quick check
```

The exporter validates inversion count and minimum Jacobian before writing,
and reports the mesh statistics (nodes, cells, wall/far faces).

## Uncertainty Quantification (per-prediction bands)

`aerowing train --ensemble K` trains a deep ensemble of K physics-regularized
surrogates with distinct seeds; `aerowing analyze --uncertainty` then reports
every prediction with a ± band (2-sigma across members):

```
aerowing train --ensemble 5                 # -> checkpoints/aerowing_ensemble.pt
aerowing analyze --wing <name> --uncertainty
```

- **The band is a real signal, not decoration**: ensemble spread ranks the
  per-point error across the aero outputs (pooled Spearman ~0.5 on held-out
  designs, stable across member-seed sets; test threshold 0.3) and — the key
  flywheel tie-in — **bands tighten as CFD labels accrue**: at the training
  exposure the learning loop actually uses (6–18 epoch fine-tune-style
  updates), 4x more labels shrink the mean aero band to ~46% of its size.
- **Members are leaner and numerous by design**: 5 members at width 128 (vs
  the flagship surrogate's 256). Capacity measurements showed wide/3-member
  ensembles disagree mostly from *overfit-to-split chaos* — leaner, more
  numerous members carry more signal and less noise in the spread.
- **The band flags extrapolation**: the same ensemble returns ~±10% on CL for
  an in-distribution wing but fat ±50% bands for a benchmark geometry far
  outside the training box — the model says "I'm guessing" instead of
  emitting confident garbage.
- **What is NOT claimed: calibration.** Measured coverage (~50% inside ±1σ,
  ~70% inside ±2σ) runs below nominal — the bands are honest about *ranking*
  and *shrinkage*, but are not probability claims.
- **Known limiter, documented as the open lever**: train members much longer
  (40+ epochs) and the spread becomes dominated by per-member overfit to
  random train/val splits — the direction can even invert. Remedy is member
  regularization (dropout / weight decay), already present in the blocks but
  effectively off (dropout 0.02) — the next tuning knob if bands need to
  keep tightening at long training.
- Cheap by design: no Bayesian machinery, no extra dependencies; each member
  is the existing physics-regularized surrogate training loop under its own
  seed (Fourier embedding, weights, train/val split all differ per member).

## Web Privacy (100% private)

* The web UI binds to `127.0.0.1` only — never reachable from the network.
* Zero third-party requests: Three.js is vendored locally, no CDN, no external
  fonts, no analytics, no telemetry, no CORS exposure.
* AI endpoints use trained models from a local checkpoint only
  (`AEROWING_CHECKPOINT`, default `checkpoints/aerowing_models.pt`); without a
  checkpoint, inverse design is refused rather than returning random output.
* All visualization data comes from the physics solvers (VLM `delta_cp`), never
  synthetic placeholders.

## Tests

```bash
pytest tests/ -v
```

## Structure

| Component       | Description |
|-----------------|-------------|
| `aerowing/geometry/` | CST 3D parameterization, wing builder, benchmarks |
| `aerowing/solvers/` | VLM, SU2, viscous, wave drag, aero engine |
| `aerowing/models/` | Dataset, generator, surrogate, trainer for ML |
| `aerowing/continual.py` | Continuous learning: data lake, CFD quality gate, replay, promotion, growth |
| `aerowing/collector.py` | Batch CFD output sweeper (zero-manual-step flywheel) |
| `aerowing/multi_fidelity.py` | Multi-fidelity study harness (error-vs-cost money slide) |
| `aerowing/mesher_3d.py` | Structured hexahedral O-grid volume mesher (SU2 output) |
| `aerowing/models/ensemble_3d.py` | Deep-ensemble UQ: per-prediction ±bands that tighten as labels accrue |
| `aerowing/optimization/` | Gradient MDO, inverse design, Pareto NSGA-II |
| `aerowing/export/` | STEP, STL, SU2, VTK exporters |
| `aerowing/web/` | FastAPI server + static frontend (100% private) |
| `aerowing/cli/` | Command-line interface |
| `aerowing/config.py` | YAML configuration loader |
| `configs/` | YAML configurations for 3D wing cases |
| `tests/` | Unit tests: geometry, VLM physics, VLM scheme properties (regression guards for the discretization signature), continual learning, collector, multi-fidelity study, mesher, ensemble UQ, exporters, optimization, web API |

See `pyproject.toml` for dependencies and `aerowing/cli/main_cli.py` for the CLI reference.