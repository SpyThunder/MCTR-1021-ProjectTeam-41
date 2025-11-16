"""
ga_optimizer.py  (ARITHMETIC RECOMBINATION)

Genetic Algorithm (real-coded) with fixed percentages:
 - Elitism fraction (default 0.20)
 - Crossover fraction (default 0.60) -> produced by Whole Arithmetic Recombination
 - Mutation fraction (default 0.20) -> exact fraction of population mutated each generation
   (NO per-gene mutation probability; mutation here means: pick an individual and
    apply Gaussian perturbation to ALL genes).

Whole Arithmetic Recombination:
 - For gene i: child[i] = alpha * parent_a[i] + (1 - alpha) * parent_b[i]
 - alpha is a parameter you set when creating the GA (default 0.5).
 - Result is clamped to [lo,hi].

Mutation:
 - For each mutated individual, for each gene i with bounds (lo,hi) we add
   Gaussian noise N(0, sigma_i) where sigma_i = mutation_scale * (hi - lo).
 - Result is clamped to [lo,hi].

Usage:
 - Place next to your PID.py and SA.py (optional) files.
 - Run: python ga_optimizer.py
"""

import random
import math
import time
import csv
import os
from copy import deepcopy

import matplotlib.pyplot as plt

# Import the PID simulation functions (adjust name/path if needed)
try:
    from PID import simulate_and_cost, visualize_pid, straight_traffic_factory
except Exception as e:
    raise ImportError("Failed to import PID module (PID.py). Make sure PID.py is in same folder. "
                      f"Original error: {e}")

# Try to import SA for comparison (optional)
_have_SA = True
try:
    from SA import simulated_annealing
except Exception:
    _have_SA = False
    print("[ga_optimizer] Warning: couldn't import SA.py → SA comparison disabled.")


class GA:
    def __init__(
        self,
        evaluate_fn,
        bounds=((0.0, 100), (0.0, 10), (0.0, 10.0)),
        pop_size=50,
        generations=100,
        elite_frac=0.20,
        crossover_frac=0.60,
        mutation_frac=0.20,
        arith_alpha=0.5,           # <-- YOU set this alpha (0..1)
        mutation_scale=0.08,       # fraction of gene range used as sigma for Gaussian mutation
        tournament_k=3,
        rng_seed=0,
        verbose=True,
    ):
        """
        GA constructor using Whole Arithmetic Recombination with user-specified alpha.
        Note: fractions ideally sum to 1.0; if they do not, they will be normalized.
        """
        self.evaluate_fn = evaluate_fn
        self.bounds = bounds
        self.pop_size = int(pop_size)
        self.generations = int(generations)

        # Normalize fractions if they don't sum to 1.0 exactly
        total = (elite_frac + crossover_frac + mutation_frac)
        if abs(total - 1.0) > 1e-12:
            elite_frac /= total
            crossover_frac /= total
            mutation_frac /= total

        self.elite_frac = float(elite_frac)
        self.crossover_frac = float(crossover_frac)
        self.mutation_frac = float(mutation_frac)

        # arithmetic recombination alpha (user-controlled)
        if not (0.0 <= arith_alpha <= 1.0):
            raise ValueError("arith_alpha must be in [0,1].")
        self.arith_alpha = float(arith_alpha)

        self.mutation_scale = float(mutation_scale)

        self.tournament_k = int(tournament_k)
        self.rng_seed = rng_seed
        self.verbose = bool(verbose)

        random.seed(self.rng_seed)

        # precompute counts (rounded each generation to nearest integer; final adjustment done per-gen)
        self._compute_counts()

    def _compute_counts(self):
        """Compute integer counts from fractions; these are used as base values and adjusted per generation."""
        N = self.pop_size
        # ensure at least 1 elite for reasonable populations
        self.elite_count = max(1, round(self.elite_frac * N))
        self.crossover_count = round(self.crossover_frac * N)
        # mutation_count is computed to ensure sum = N (adjusted later if needed)
        self.mutation_count = N - self.elite_count - self.crossover_count
        # if rounding made mutation_count negative, fix by reducing crossover_count
        if self.mutation_count < 0:
            deficit = -self.mutation_count
            self.crossover_count = max(0, self.crossover_count - deficit)
            self.mutation_count = N - self.elite_count - self.crossover_count
        # Final safety clamps
        self.elite_count = max(0, min(N, self.elite_count))
        self.crossover_count = max(0, min(N - self.elite_count, self.crossover_count))
        self.mutation_count = N - self.elite_count - self.crossover_count

    def _init_population(self):
        pop = []
        for _ in range(self.pop_size):
            indiv = [random.uniform(lo, hi) for (lo, hi) in self.bounds]
            pop.append(indiv)
        return pop

    def _evaluate_population(self, pop, other_cars_fn=None):
        costs = []
        for indiv in pop:
            cost, _ = self.evaluate_fn(*indiv, other_cars_fn=other_cars_fn)
            costs.append(cost)
        return costs

    def _tournament_select(self, pop, costs):
        best = None
        best_cost = float("inf")
        for _ in range(self.tournament_k):
            i = random.randrange(len(pop))
            if costs[i] < best_cost:
                best_cost = costs[i]
                best = pop[i]
        return deepcopy(best)

    def _arith_recomb_child(self, parent_a, parent_b):
        """Whole Arithmetic Recombination (user alpha) per-gene; clamp to bounds."""
        child = []
        a_coef = self.arith_alpha
        b_coef = 1.0 - a_coef
        for i, (lo, hi) in enumerate(self.bounds):
            val = a_coef * parent_a[i] + b_coef * parent_b[i]
            # clamp
            val = max(lo, min(hi, val))
            child.append(val)
        return child

    def _mutate_whole(self, parent):
        """
        Deterministic mutation for an individual:
         - Applies Gaussian perturbation to ALL genes (no per-gene probability).
         - sigma_i = mutation_scale * (hi - lo)
         - clamp to bounds
        """
        child = parent[:]
        for i, (lo, hi) in enumerate(self.bounds):
            sigma = self.mutation_scale * (hi - lo)
            child[i] = child[i] + random.gauss(0, sigma)
            child[i] = max(lo, min(hi, child[i]))
        return child

    def optimize(self, initial_population=None, other_cars_fn=None, return_history=True,
                 live_plot=True, plot_every=10):
        """
        Main optimization loop.

        Parameters:
         - initial_population: optional list of individuals to seed the population
         - other_cars_fn: passed to evaluate_fn for collision testing
         - return_history: return detailed history dict if True
         - live_plot: if True, show a single persistent convergence plot updated every `plot_every` generations
         - plot_every: update frequency (int). Set to 10 to update every 10 generations.

        Returns:
         - dict with best_gains, best_cost, history
        """
        # recompute counts in case pop_size changed externally
        self._compute_counts()
        N = self.pop_size
        E = self.elite_count
        C = self.crossover_count
        M = self.mutation_count

        # logging of exact counts & percentages
        if self.verbose:
            print(f"[GA init] pop_size={N} elites={E} ({E/N*100:.2f}%)  "
                  f"crossover_children={C} ({C/N*100:.2f}%)  "
                  f"mutation_children={M} ({M/N*100:.2f}%)  "
                  f"arith_alpha={self.arith_alpha}  mutation_scale={self.mutation_scale}")

        if initial_population is None:
            population = self._init_population()
        else:
            population = [deepcopy(ind) for ind in initial_population]
            while len(population) < N:
                population.append([random.uniform(lo, hi) for (lo, hi) in self.bounds])

        costs = self._evaluate_population(population, other_cars_fn=other_cars_fn)
        history = {"gen": [], "best_cost": [], "mean_cost": [], "best_gains": []}

        # initial best
        best_idx = int(min(range(len(costs)), key=lambda i: costs[i]))
        best = deepcopy(population[best_idx])
        best_cost = costs[best_idx]

        # Setup live plot if requested
        if live_plot:
            plt.ion()
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.set_title("GA Convergence (best & mean cost)")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Cost (lower = better)")
            ax.grid(True)
            line_best, = ax.plot([], [], label="Best (so far)", lw=2)
            line_mean, = ax.plot([], [], label="Mean", lw=1, linestyle="--", alpha=0.8)
            ax.legend()

        for gen in range(self.generations):
            # recalc counts in case rounding issues; ensure sums
            self._compute_counts()
            E = self.elite_count
            C = self.crossover_count
            M = self.mutation_count

            new_pop = []

            # 1) Elitism: copy top E individuals unchanged
            sorted_idx = sorted(range(len(population)), key=lambda i: costs[i])
            for e in range(E):
                new_pop.append(deepcopy(population[sorted_idx[e]]))

            # 2) Crossover children: create exactly C children using tournament selection + arithmetic recombination
            for _ in range(C):
                p1 = self._tournament_select(population, costs)
                p2 = self._tournament_select(population, costs)
                child = self._arith_recomb_child(p1, p2)
                new_pop.append(child)

            # 3) Mutation children: create exactly M children by selecting parents and applying whole-individual mutation
            for _ in range(M):
                p = self._tournament_select(population, costs)
                child = self._mutate_whole(p)
                new_pop.append(child)

            # Final safety: if rounding caused a mismatch, trim or fill randomly
            if len(new_pop) > N:
                new_pop = new_pop[:N]
            while len(new_pop) < N:
                # Fill with random individuals (shouldn't usually be needed)
                new_pop.append([random.uniform(lo, hi) for (lo, hi) in self.bounds])

            population = new_pop
            costs = self._evaluate_population(population, other_cars_fn=other_cars_fn)

            # Update best
            best_idx = int(min(range(len(costs)), key=lambda i: costs[i]))
            gen_best_cost = costs[best_idx]
            gen_best = deepcopy(population[best_idx])
            mean_cost = sum(costs) / len(costs)

            if gen_best_cost < best_cost:
                best_cost = gen_best_cost
                best = deepcopy(gen_best)

            history["gen"].append(gen)
            history["best_cost"].append(best_cost)
            history["mean_cost"].append(mean_cost)
            history["best_gains"].append(tuple(best))

            # Live plot update every plot_every generations (on same figure)
            if live_plot and ((gen % plot_every == 0 and gen > 0) or gen == self.generations - 1):
                line_best.set_xdata(list(range(len(history["gen"]))))
                line_best.set_ydata(history["best_cost"])
                line_mean.set_xdata(list(range(len(history["gen"]))))
                line_mean.set_ydata(history["mean_cost"])
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.001)

            if self.verbose and (gen == 0 or gen % max(1, self.generations // 10) == 0 or gen == self.generations - 1):
                print(f"[GA] gen={gen:4d} best_cost={best_cost:.6f} mean_cost={mean_cost:.6f} best_gains={best}")

        # finalize plot
        if live_plot:
            plt.ioff()
            # ensure final data is shown
            line_best.set_xdata(list(range(len(history["gen"]))))
            line_best.set_ydata(history["best_cost"])
            line_mean.set_xdata(list(range(len(history["gen"]))))
            line_mean.set_ydata(history["mean_cost"])
            ax.relim()
            ax.autoscale_view()
            plt.show()

        result = {"best_gains": tuple(best), "best_cost": best_cost, "history": history}
        if return_history:
            return result
        else:
            return {"best_gains": tuple(best), "best_cost": best_cost}


# ---------------------------
# Utilities (unchanged)
# ---------------------------
def plot_convergence(history_ga, history_sa=None, title="Fitness convergence", savefile=None):
    plt.figure(figsize=(8, 4))
    plt.plot(history_ga["gen"], history_ga["best_cost"], label="GA best")
    plt.plot(history_ga["gen"], history_ga["mean_cost"], label="GA mean", alpha=0.6)
    if history_sa is not None:
        gens_sa = [x[0] for x in history_sa]
        bests_sa = [x[2] for x in history_sa]
        # match lengths
        L = min(len(history_ga["gen"]), len(bests_sa))
        plt.plot(history_ga["gen"][:L], bests_sa[:L], label="SA best", linestyle="--")
    plt.yscale("symlog")
    plt.xlabel("Generation / Iteration")
    plt.ylabel("Cost (lower = better)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if savefile:
        plt.savefig(savefile, dpi=150)
    plt.show()


def plot_cte_comparison(gains_list, labels, path_params=(40, 4, 300), dt=0.025):
    plt.figure(figsize=(8, 4))
    for gains, label in zip(gains_list, labels):
        Kp, Ki, Kd = gains
        cost, cte = simulate_and_cost(Kp, Ki, Kd, dt=dt, path_params=path_params)
        if len(cte) == 0:
            print(f"[plot_cte] {label} produced empty CTE (likely out-of-bounds).")
            continue
        plt.plot(cte, label=f"{label} (cost {cost:.2f})", alpha=0.9)
    plt.xlabel("Timestep")
    plt.ylabel("Cross-track error (m)")
    plt.title("CTE over time")
    plt.legend()
    plt.grid(True)
    plt.show()


def save_history_csv(history, filename):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gen", "best_cost", "mean_cost", "best_Kp", "best_Ki", "best_Kd"])
        for i, gen in enumerate(history["gen"]):
            kp, ki, kd = history["best_gains"][i]
            w.writerow([gen, history["best_cost"][i], history["mean_cost"][i], kp, ki, kd])
    print(f"[save_history_csv] Written {filename}")


# ---------------------------
# Case runner (uses the new GA)
# ---------------------------
def run_case_study(case_name, path_params=(40, 4, 300), other_cars_fn=None,
                   ga_params=None, sa_params=None, compare_to_sa=True):
    print(f"\n=== Running case: {case_name} ===")
    if ga_params is None:
        ga_params = {}

    ga = GA(evaluate_fn=simulate_and_cost, verbose=True, **ga_params)

    t0 = time.time()
    # enable live_plot and set plot_every=10 by default
    ga_result = ga.optimize(other_cars_fn=other_cars_fn, live_plot=True, plot_every=10)
    t_ga = time.time() - t0
    print(f"[GA] Done. Best gains: {ga_result['best_gains']}  cost={ga_result['best_cost']:.6f}  time={t_ga:.2f}s")

    sa_result = None
    if compare_to_sa and _have_SA:
        sa_params = sa_params or {}
        sa_short_params = dict(
            initial_gains=(0.5, 0.05, 0.1),
            initial_temp=1.0,
            cooling_rate=0.995,
            iterations=1000,
            step_scale=(20, 2, 5),
            param_bounds=((0.0, 100), (0.0, 10), (0.0, 10)),
            evaluate_fn=simulate_and_cost,
            verbose_every=200,
            visualize_every=None,
            visualize_blocking=False,
            other_cars_fn=other_cars_fn,
        )
        sa_short_params.update(sa_params)
        print("[GA] Running SA (short) for comparison...")
        t0 = time.time()
        sa_result = simulated_annealing(**sa_short_params)
        t_sa = time.time() - t0
        print(f"[SA] Done. Best gains: {sa_result['best_gains']} cost={sa_result['best_cost']:.6f} time={t_sa:.2f}s")
    else:
        print("[GA] SA comparison skipped (SA.py not available).")

    plot_convergence(ga_result["history"], history_sa=sa_result["history"] if sa_result else None,
                     title=f"Convergence - {case_name}")

    baseline = (0.05, 0.0005, 0.15)
    gains_list = [ga_result["best_gains"]]
    labels = ["GA best"]
    if sa_result:
        gains_list.append(sa_result["best_gains"])
        labels.append("SA best")
    gains_list.append(baseline)
    labels.append("Baseline (default)")
    plot_cte_comparison(gains_list, labels, path_params=path_params)

    outdir = "ga_results"
    os.makedirs(outdir, exist_ok=True)
    csvname = os.path.join(outdir, f"history_{case_name.replace(' ', '_')}.csv")
    save_history_csv(ga_result["history"], csvname)

    return {"ga": ga_result, "sa": sa_result}


# ---------------------------
# Default suite (adjusted to the new fraction interface)
# ---------------------------
def default_suite():
    results = {}

    # Case 1: easy
    results["case_easy"] = run_case_study(
        "Easy - default path, no traffic",
        path_params=(40, 4, 300),
        other_cars_fn=None,
        ga_params={
            "pop_size": 50,
            "generations": 80,
            "elite_frac": 0.20,
            "crossover_frac": 0.60,
            "mutation_frac": 0.20,
            "arith_alpha": 0.5,           # you set this
            "mutation_scale": 0.08,
            "tournament_k": 3,
            "rng_seed": 0,
        },
        compare_to_sa=_have_SA,
    )

    # Case 2: harder
    results["case_harder"] = run_case_study(
        "Harder - larger lateral amplitude",
        path_params=(40, 8, 300),
        other_cars_fn=None,
        ga_params={
            "pop_size": 60,
            "generations": 120,
            "elite_frac": 0.20,
            "crossover_frac": 0.60,
            "mutation_frac": 0.20,
            "arith_alpha": 0.5,
            "mutation_scale": 0.10,
            "tournament_k": 3,
            "rng_seed": 1,
        },
        compare_to_sa=_have_SA,
    )

    # Case 3: collisions
    other_fn = straight_traffic_factory(v=4.5, lane_y=0.0, n=2)
    results["case_collision"] = run_case_study(
        "Collision - traffic present",
        path_params=(40, 4, 300),
        other_cars_fn=other_fn,
        ga_params={
            "pop_size": 70,
            "generations": 140,
            "elite_frac": 0.20,
            "crossover_frac": 0.60,
            "mutation_frac": 0.20,
            "arith_alpha": 0.5,
            "mutation_scale": 0.10,
            "tournament_k": 4,
            "rng_seed": 2,
        },
        compare_to_sa=_have_SA,
    )

    return results


# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    tstart = time.time()
    print("Running GA optimizer suite with fixed percents (this may take a few minutes)...")
    results = default_suite()
    print("\nAll case studies finished in %.2f s" % (time.time() - tstart))
    for k, v in results.items():
        print(f"{k}: GA best={v['ga']['best_gains']} cost={v['ga']['best_cost']:.6f}")
        if v["sa"] is not None:
            print(f"     SA best={v['sa']['best_gains']} cost={v['sa']['best_cost']:.6f}")
    print("Results saved in ./ga_results/*.csv")
