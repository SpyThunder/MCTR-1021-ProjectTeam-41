import random
import math
import time
import csv
import os
from copy import deepcopy

import matplotlib.pyplot as plt

from SA import simulated_annealing
from GA import genetic_algorithm
from PID import simulate_and_cost, visualize_pid, straight_traffic_factory

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


def plot_cte_comparison(gains_list, labels,  dt=0.025):
    plt.figure(figsize=(8, 4))
    for gains, label in zip(gains_list, labels):
        Kp, Ki, Kd = gains
        cost, cte = simulate_and_cost(Kp, Ki, Kd, dt=dt, path_type=1)
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


def run_case_study(case_name, path_params=(40, 4, 300), other_cars_fn=None,
                   ga_params=None, sa_params=None,
                   compare_to_sa=True,
                   ga_runs=5, sa_runs=5):

    print(f"\n=== Running case: {case_name} ===")

    if ga_params is None:
        ga_params = {}

    # ---- GA MULTI-RUN ----
    print(f"[GA] Running {ga_runs} times...")
    t0 = time.time()
    ga_multi = run_ga_multiple(ga_runs, ga_params, other_cars_fn)
    t_ga = time.time() - t0

    print("\n[GA] SUMMARY")
    print(f"  Mean cost: {ga_multi['mean']:.4f}")
    print(f"  Std cost : {ga_multi['std']:.4f}")
    print(f"  Best cost: {min(ga_multi['best_costs']):.4f}")
    print(f"  Best gains: {ga_multi['best_overall']}")
    print(f"  Time: {t_ga:.2f}s")

    sa_multi = None
    if compare_to_sa:
        print(f"\n[SA] Running {sa_runs} times...")
        t0 = time.time()
        sa_multi = run_sa_multiple(sa_runs, sa_params or {}, other_cars_fn)
        t_sa = time.time() - t0

        print("\n[SA] SUMMARY")
        print(f"  Mean cost: {sa_multi['mean']:.4f}")
        print(f"  Std cost : {sa_multi['std']:.4f}")
        print(f"  Best cost: {min(sa_multi['best_costs']):.4f}")
        print(f"  Best gains: {sa_multi['best_overall']}")
        print(f"  Time: {t_sa:.2f}s")

    # Plot using the FIRST run history
    history_ga_raw = ga_multi["histories"][0]
    history_ga = {
        "gen": [x[0] for x in history_ga_raw],
        "best_cost": [x[1] for x in history_ga_raw],
        "mean_cost": [x[1] for x in history_ga_raw],
        "best_gains": [x[2] for x in history_ga_raw],
    }

    history_sa = sa_multi["histories"][0] if sa_multi else None

    plot_convergence(history_ga, history_sa, title=f"Convergence - {case_name}")

    # Compare best runs
    gains_list = [ga_multi["best_overall"]]
    labels = ["GA best over runs"]

    if sa_multi:
        gains_list.append(sa_multi["best_overall"])
        labels.append("SA best over runs")


    plot_cte_comparison(gains_list, labels)

    return {"ga_multi": ga_multi, "sa_multi": sa_multi}




# ---------------------------
# Default suite (adjusted to the new fraction interface)
# ---------------------------
def default_suite():
    results = {}

    # Case 3: collisions
    other_fn = straight_traffic_factory(v=4.5, lane_y=0.0, n=2)
    results["case_collision"] = run_case_study(
        "Collision - traffic present",
        path_params=(40, 4, 300),
        other_cars_fn=other_fn,
        ga_params={
            "pop_size": 100,
            "generations": 100,
            "elite_frac": 0.20,
            "crossover_frac": 0.60,
            "mutation_frac": 0.20,
            "arith_alpha": 0.5,
            "mutation_scale": (20, 2, 5),  # FIXED
            "tournament_k": 4,
            "rng_seed": 2,
        },
        compare_to_sa=True,
    )

    return results

import numpy as np

def run_ga_multiple(times, ga_params, other_cars_fn=None):
    best_costs = []
    best_gains = []
    histories = []

    for i in range(times):
        print(f"\n[GA] Multi-run {i+1}/{times}")
        mapped = dict(
            generations       = ga_params.get("generations", 100),
            population_size   = ga_params.get("pop_size", 100),
            crossover_alpha   = ga_params.get("arith_alpha", 0.5),
            elitism_ratio     = ga_params.get("elite_frac", 0.2),
            crossover_ratio   = ga_params.get("crossover_frac", 0.6),
            mutation_ratio    = ga_params.get("mutation_frac", 0.2),
            mutation_scale    = ga_params.get("mutation_scale", (20, 2, 5)),
            visualize_every   = None,
            visualize_blocking=False,
            other_cars_fn     = other_cars_fn,
            rng_seed = i
        )
        out = genetic_algorithm(**mapped)
        best_costs.append(out["best_cost"])
        best_gains.append(out["best_gains"])
        histories.append(out["history"])

    return {
        "best_costs": best_costs,
        "best_gains": best_gains,
        "histories": histories,
        "mean": float(np.mean(best_costs)),
        "std": float(np.std(best_costs)),
        "best_overall": best_gains[np.argmin(best_costs)],
    }


def run_sa_multiple(times, sa_params, other_cars_fn=None):
    best_costs = []
    best_gains = []
    histories = []

    for i in range(times):
        print(f"\n[SA] Multi-run {i+1}/{times}")
        p = dict(
            initial_gains=(0.5, 0.05, 0.1),
            initial_temp=1.0,
            cooling_rate=0.995,
            iterations=10000,
            step_scale=(20, 2, 5),
            param_bounds=((0.0, 100), (0.0, 10), (0.0, 10)),
            evaluate_fn=simulate_and_cost,
            verbose_every=200,
            visualize_every=None,
            visualize_blocking=False,
            other_cars_fn=other_cars_fn,
            rng_seed = i
        )
        p.update(sa_params)

        out = simulated_annealing(**p)
        best_costs.append(out["best_cost"])
        best_gains.append(out["best_gains"])
        histories.append(out["history"])

    return {
        "best_costs": best_costs,
        "best_gains": best_gains,
        "histories": histories,
        "mean": float(np.mean(best_costs)),
        "std": float(np.std(best_costs)),
        "best_overall": best_gains[np.argmin(best_costs)],
    }
import numpy as np
import matplotlib.pyplot as plt

def plot_gaussian(mean, std, label="Distribution", color=None, shade=True):
    """
    Plot a Gaussian (normal) PDF from mean and standard deviation.
    """
    if std == 0:
        # Special case: degenerate distribution (spike at the mean)
        x = np.linspace(mean - 1, mean + 1, 400)
        y = np.zeros_like(x)
        center_idx = np.argmin(np.abs(x - mean))
        y[center_idx] = 1.0  # spike
        plt.plot(x, y, label=f"{label} (std=0)", linewidth=2, color=color)
        return

    # Normal distribution range
    x = np.linspace(mean - 4 * std, mean + 4 * std, 400)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    plt.plot(x, y, label=f"{label} (μ={mean:.3f}, σ={std:.3f})", linewidth=2, color=color)

    if shade:
        plt.fill_between(x, y, alpha=0.2, color=color)

def plot_gaussian_comparison(ga_mean, ga_std, sa_mean=None, sa_std=None):
    plt.figure(figsize=(8, 4))

    # GA distribution
    plot_gaussian(ga_mean, ga_std, label="GA Distribution", color="blue")

    # SA distribution (if provided)
    if sa_mean is not None and sa_std is not None:
        plot_gaussian(sa_mean, sa_std, label="SA Distribution", color="red")

    plt.title("Gaussian Distribution of Best Costs Across Runs")
    plt.xlabel("Cost")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    tstart = time.time()
    print("Running genetic_algorithm optimizer suite with fixed percents (this may take a few minutes)...")
    results = default_suite()
    print("\nAll case studies finished in %.2f s" % (time.time() - tstart))
    for k, v in results.items():
        ga = v["ga_multi"]
        ga_mean = ga["mean"]
        ga_std = ga["std"]
        print(f"{k}: GA mean={ga['mean']:.6f} std={ga['std']:.6f} best={min(ga['best_costs']):.6f}")
        print(f"     GA best gains: {ga['best_overall']}")

        if v["sa_multi"] is not None:
            sa = v["sa_multi"]
            sa_mean = sa["mean"]
            sa_std = sa["std"]
            print(f"     SA mean={sa['mean']:.6f} std={sa['std']:.6f} best={min(sa['best_costs']):.6f}")
            print(f"     SA best gains: {sa['best_overall']}")
    plot_gaussian_comparison(ga_mean, ga_std, sa_mean, sa_std)

