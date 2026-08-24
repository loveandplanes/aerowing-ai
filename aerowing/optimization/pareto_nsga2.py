"""
Multi-Objective Pareto Optimization Suite (NSGA-II) for 3D Wing MDO.
Balances Cruise Aerodynamic Efficiency (L/D), Fuel Tank Capacity, and Root Bending Moment.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from ..geometry.wing_3d import Wing3D
from ..models.surrogate_3d import AeroSurrogate3D


class Individual3D:
    """An individual 3D wing design candidate."""
    def __init__(self, params: np.ndarray):
        self.params = np.asarray(params, dtype=float)
        self.objectives = np.zeros(3)  # [-L/D, -FuelVolume, BendingMoment]
        self.constraints_violation = 0.0
        self.rank = 0
        self.crowding_distance = 0.0
        self.telemetry = {}
        self.domination_count = 0
        self.dominated_set = []


class ParetoOptimizerNSGA2:
    """
    Non-Dominated Sorting Genetic Algorithm II (NSGA-II) for 3D Aerospace Wing Optimization.
    """

    def __init__(
        self,
        surrogate: AeroSurrogate3D,
        target_cl: float = 0.55,
        mach: float = 0.82,
        reynolds: float = 2.5e7,
        pop_size: int = 40,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.85,
    ):
        self.surrogate = surrogate
        self.target_cl = target_cl
        self.mach = mach
        self.reynolds = reynolds
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Planform bounds: [span, AR, taper, sweep, dihedral, twist_r, twist_t]
        self.bounds_min = np.array([20.0, 6.0, 0.20, 10.0, 1.0, 0.0, -4.5])
        self.bounds_max = np.array([45.0, 13.0, 0.60, 35.0, 5.0, 3.5, 0.0])

    def _evaluate(self, ind: Individual3D):
        """Evaluates objectives using the fast AI surrogate (< 1ms)."""
        res = self.surrogate.predict_wing(
            ind.params,
            alpha_deg=2.5,
            mach=self.mach,
            reynolds=self.reynolds,
        )
        ind.telemetry = res

        # Objective 1: Maximize L/D (minimize -L/D)
        obj_ld = -res["l_over_d"]

        # Objective 2: Maximize Fuel Volume (minimize -Fuel)
        obj_fuel = -res["fuel_volume_m3"]

        # Objective 3: Minimize Root Bending Moment ~ CL * Span * 0.45
        span = ind.params[0]
        root_moment = res["cl"] * (span / 2.0) * 0.45
        obj_moment = root_moment

        ind.objectives = np.array([obj_ld, obj_fuel, obj_moment])

        # Constraint: CL >= target_cl
        viol = max(0.0, self.target_cl - res["cl"])
        ind.constraints_violation = viol

    def _fast_non_dominated_sort(self, population: List[Individual3D]) -> List[List[Individual3D]]:
        """Sorts population into Pareto dominance fronts."""
        fronts = [[]]
        for p in population:
            p.domination_count = 0
            p.dominated_set = []
            for q in population:
                # p dominates q if p is <= q in all objectives and < in at least one
                if (all(p.objectives <= q.objectives) and any(p.objectives < q.objectives)) and p.constraints_violation <= q.constraints_violation:
                    p.dominated_set.append(q)
                elif (all(q.objectives <= p.objectives) and any(q.objectives < p.objectives)) and q.constraints_violation <= p.constraints_violation:
                    p.domination_count += 1

            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_set:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

    def _crowding_distance_assignment(self, front: List[Individual3D]):
        """Calculates crowding distance for diversity maintenance."""
        l = len(front)
        if l == 0:
            return
        for ind in front:
            ind.crowding_distance = 0.0

        num_objectives = len(front[0].objectives)
        for m in range(num_objectives):
            front.sort(key=lambda ind: ind.objectives[m])
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            obj_range = front[-1].objectives[m] - front[0].objectives[m]
            if obj_range == 0:
                continue

            for i in range(1, l - 1):
                front[i].crowding_distance += (
                    (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / obj_range
                )

    def optimize(self, generations: int = 25, initial_wing: Optional[Wing3D] = None) -> List[Individual3D]:
        """Runs the NSGA-II evolutionary optimization loop."""
        # 1. Initialize population
        base_vec = initial_wing.to_parameter_vector() if initial_wing else Wing3D().to_parameter_vector()
        pop = []
        for _ in range(self.pop_size):
            p = base_vec.copy()
            # Randomize planform parameters within bounds
            p[:7] = np.random.uniform(self.bounds_min, self.bounds_max)
            ind = Individual3D(p)
            self._evaluate(ind)
            pop.append(ind)

        # 2. Main Generation Loop
        for gen in range(generations):
            # Tournament selection + Simulated Binary Crossover (SBX) + Polynomial Mutation
            offspring = []
            while len(offspring) < self.pop_size:
                # Tournament
                p1, p2 = np.random.choice(pop, size=2, replace=False)
                parent1 = p1 if p1.rank < p2.rank or (p1.rank == p2.rank and p1.crowding_distance > p2.crowding_distance) else p2
                p3, p4 = np.random.choice(pop, size=2, replace=False)
                parent2 = p3 if p3.rank < p4.rank or (p3.rank == p4.rank and p3.crowding_distance > p4.crowding_distance) else p4

                # Crossover
                c1_params = parent1.params.copy()
                c2_params = parent2.params.copy()
                if np.random.rand() < self.crossover_rate:
                    alpha = np.random.rand(len(c1_params))
                    c1_params = alpha * parent1.params + (1 - alpha) * parent2.params
                    c2_params = (1 - alpha) * parent1.params + alpha * parent2.params

                # Mutation
                for child_params in [c1_params, c2_params]:
                    if np.random.rand() < self.mutation_rate:
                        mut = np.random.normal(0, 0.05, size=child_params.shape)
                        child_params += mut
                    # Clip planform bounds
                    child_params[:7] = np.clip(child_params[:7], self.bounds_min, self.bounds_max)

                    ind = Individual3D(child_params)
                    self._evaluate(ind)
                    offspring.append(ind)

            # Combine and Select Best Fronts
            combined = pop + offspring[:self.pop_size]
            fronts = self._fast_non_dominated_sort(combined)
            new_pop = []
            for front in fronts:
                self._crowding_distance_assignment(front)
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(front)
                else:
                    front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                    new_pop.extend(front[: self.pop_size - len(new_pop)])
                    break
            pop = new_pop

        # Return Rank 0 (Pareto Frontier)
        fronts = self._fast_non_dominated_sort(pop)
        return fronts[0] if len(fronts) > 0 else pop
