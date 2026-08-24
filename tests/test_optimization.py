"""
Optimization unit tests: NSGA-II Pareto front and autograd gradient MDO.
"""

import numpy as np

from aerowing.models.surrogate_3d import AeroSurrogate3D
from aerowing.optimization.pareto_nsga2 import ParetoOptimizerNSGA2, Individual3D
from aerowing.optimization.gradient_mdo import GradientOptimizer3D
from aerowing.geometry.wing_3d import Wing3D


def test_individual_initialized_attributes():
    """Individual3D declares NSGA-II bookkeeping attributes up front."""
    ind = Individual3D(np.zeros(37))
    assert ind.domination_count == 0
    assert ind.dominated_set == []
    assert ind.rank == 0
    assert ind.crowding_distance == 0.0


def test_nsga2_returns_pareto_front():
    surrogate = AeroSurrogate3D()
    opt = ParetoOptimizerNSGA2(surrogate, pop_size=10)
    frontier = opt.optimize(generations=3)
    assert len(frontier) > 0
    assert all(ind.rank == 0 for ind in frontier)


def test_gradient_mdo_autograd_runs():
    """SQP with analytic autograd Jacobian produces a valid result."""
    surrogate = AeroSurrogate3D()
    opt = GradientOptimizer3D(surrogate, target_cl=0.55, mach=0.82)
    result = opt.optimize(Wing3D(), max_iter=3)
    assert "optimized_wing" in result
    assert "optimized_params" in result
    assert "telemetry" in result
    assert np.all(np.isfinite(result["optimized_params"]))