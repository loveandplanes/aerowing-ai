# Ideas Behind the Growing AI

> How a network *learns while it grows* determines whether new data
> corrects previous knowledge, overwrites it, or refines it — and how its
> internal "thought" organizes across dimensions, how efficiently it
> adapts, and how accurately it ends.

This document collects the ideas explored in this repository. They were
conceived independently, without prior knowledge of much of the literature
cited below. Because this work was not built *from* that literature,
references have been traced afterwards — some may still be missing, and
corrections are welcome.

---

## 1. Why growth-time learning matters

A network that grows is not just a bigger network. It is a network at a
bifurcation point: at the instant of growth its function is preserved
exactly, but the next hundred gradient steps decide *what the new capacity
will become responsible for*.

That decision has four consequences:

1. **Correction vs. overwriting.** Does new data *correct* previous
   knowledge where it was wrong, or does it *overwrite* where it was right?
2. **What is learned.** Does the network learn the basic physics again from
   scratch, or does it learn the *physics of the correction* — the residual
   between what it knew and what the new data demands?
3. **How thought organizes across dimensions.** In a physical problem,
   dimensions have meaning (spanwise position, chordwise pressure, Mach
   regime). Growth can either smear new knowledge globally or localize it
   where it belongs.
4. **Efficiency vs. accuracy.** A network trained from scratch on all data
   with the same final parameter count will *likely beat* one trained by
   addition on raw accuracy alone. That is expected and documented in this
   repo. Growth is not chosen to win a benchmark — it is chosen to **control
   the network's thinking** and to **avoid paying for expensive CFD all over
   again**. Each anchor run costs hours; the question is how much accuracy
   each new run buys, not the final ceiling.

Understanding this trade-off — and measuring it honestly, including the
cases where growth loses — is the central contribution of the experiments
in `experiments/`.

---

## 2. The ideas, in plain language

### 2.1 Parallel-compare protocol
*Keep the old network frozen, build an expanded copy, and race both
against plain fine-tuning.* This is the experimental framework everything
else runs inside. Without it, you cannot tell whether growth helped or
the extra data did.

### 2.2 Conditional escalation — plasticity must be earned
*Only touch the frozen base when the new module has proven it helps,
and scale the change by how much it helped.* Three gates must open:
real learning happened, progress has plateaued, and error is still
material. Result: safe in both easy and hard regimes; thresholds remain
tunable. Tested in `conditional_expansion.py`.

### 2.3 Graded radial unfreezing
*Wake the nodes closest to the new weights first with small directed
noise; only if learning happens, wake a wider circle more quietly.*
A ripple of plasticity moving outward from the growth site. Wins when
the new task needs feature directions the old data never exercised.
Tested in `graded_expansion.py` and `..._v2.py`.

### 2.4 Difference-map gate initialization
*Compare new data's footprint vs the old average, and pre-point the new
gate where they differ most.* At growth time the gate is pre-trained on
stream-membership labels (new = 1, old = 0) — pure data geometry, no
targets. Result: 2× faster adaptation, same final quality.
Tested in `gate_map_init.py`.

### 2.5 Shift-coupled loss exponent
*When new data is very different, punish big errors harder (high
exponent); when it's similar, be gentle and catch subtle details (low
exponent).* Loss `L = mean(|error|^p)` with `p` set from measured
distribution shift. Beats every fixed exponent on the noisy regime.
Tested in `adaptive_exponent.py`.

### 2.6 Zoned error weighting and composition
*Split the input space into zones from the map, compute error per zone
in parallel vs the global error, and weight zones by their estimated
shift.* Alone: mild retention gain. Composed with the gated residual:
**best on both axes** (retain 0.181, learn 0.074). The composition is
the finding.
Tested in `zoned_loss.py`.

### 2.7 Distribution weights (Bayesian selective plasticity)
*Don't store a single number per weight — store a distribution
(mean + deviation) that shrinks as it learns.* When deviation is small,
just use the mean: the network "thinks more abstractly." Each weight
learns its own uncertainty; KL divergence anchors it to its stage-1
posterior. Best retention of any arm (0.043). Tested in
`bayes_plasticity.py`.

---

## 3. Provenance — honest accounting

These ideas — the general research questions, the conceptual direction,
and each of the seven proposals above — were conceived by the author
without AI generation of the ideas themselves and without prior knowledge
of much of the literature below. Tracing was done *afterwards* by
searching arXiv / Scholar, with AI assistance for discovery. The list is
to our knowledge complete for the mechanisms as implemented, but because
this work was not built from these papers, references may still be
missing. Pointers to closer prior art are welcome and will be added.

AI was used throughout the project as a collaborator for coding,
writing assistance, literature discovery, and to build and run the
experiments quickly — including implementing the harnesses, fixing bugs,
and drafting documentation. The general ideas and research direction
originate entirely from the author; AI accelerated their translation into
tested artifacts.

*   **Net2Net** (Chen et al., 2015) — function-preserving transformation.
*   **Self-Expanding Neural Networks** (arXiv:2307.04526) — natural-gradient
    expansion score.
*   **GradMax** (Evci et al., 2022) — gradient-maximizing initialization.
*   **GrowNNs — manifold folding at mistakes** — growth at clustered
    mispredictions with protected fine-tuning.
*   **Progressive Networks** (Rusu et al., 2016) — parallel frozen columns.
*   **GEM / A-GEM** (Lopez-Paz 2017; Chaudhry 2018) and **OGD** —
    gradient steering away from old samples.
*   **EDN** (Perrett et al., 2022) — error-driven neurogenesis.
*   **Bayes by Backprop** (Blundell et al., 2015) and **VCL**
    (Nguyen et al., 2018) — weights as distributions, KL anchoring.

Where this repository differs is the *integration*: quality-gated streaming
CFD truth, delivery masks, holdout promotion, and a controlled comparison
showing *where* structural protection wins, *where* loss shaping fails,
and *why* — in physics-surrogate regression, a setting the literature
above rarely touches.

---

## 4. Why growth if from-scratch can win on accuracy?

Because the metric that matters in industry is not final accuracy at
fixed parameter count — it is **accuracy per expensive CFD run** and
**control over what the network forgets**.

In our measurements, a network trained from scratch on all data often
beats an incrementally grown one on raw RMSE. That is expected: the
grown network carries an architectural and optimization handicap. Growth
is chosen anyway for two reasons this repo measures:

1. **It avoids paying for CFD again.** The base was trained on data
   already paid for; growth reuses it. From-scratch retraining is only
   cheaper on paper — in practice it means re-collecting and re-labeling
   the old corpus.
2. **It makes the network's thinking controllable.** With a frozen base
   and a gated correction, you can *read* responsibility (gate maps),
   *localize* corrections (zoned weighting), and *schedule* plasticity
   (graded unfreezing, shift-coupled exponents) — all demonstrated in
   `experiments/`. From-scratch training offers no such levers.

The honest trade is therefore: a small, measurable accuracy cost for
auditable control and incremental cost. For the preliminary-design loop
this repository targets — where CFD is the scarce resource — that trade
is the right one.

---

*If you know of closer prior art for any idea above, please open an
issue or pull request. Credit will be added immediately.*
