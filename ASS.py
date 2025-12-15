import random
import math
import time
import csv
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from SA import simulated_annealing
from GA import genetic_algorithm
from PID import simulate_and_cost, visualize_pid, straight_traffic_factory


# ---------------------------
# Utilities (fixed & updated)
# ---------------------------
def plot_convergence(history_ga, history_sa=None, history_pso=None, history_tlbo=None,
                     title="Fitness convergence", savefile=None):
    """
    Plot the best-so-far cost curves for GA, SA, PSO, and TLBO.
    Uses the actual 'best' fields stored in each algorithm's history.
    """
    plt.figure(figsize=(10, 6))

    # GA history (history_ga expected to be a dict with 'gen' and 'best_cost' lists)
    plt.plot(history_ga["gen"], history_ga["best_cost"], label="GA best", linewidth=2, marker='o', markersize=3)

    # SA history (list of tuples (it, current_cost, best_cost, T))
    if history_sa is not None:
        gens_sa = [x[0] for x in history_sa]
        bests_sa = [x[2] for x in history_sa]
        L = min(len(history_ga["gen"]), len(bests_sa))
        plt.plot(history_ga["gen"][:L], bests_sa[:L], label="SA best", linestyle="--", linewidth=2, marker='s',
                 markersize=3)

    # PSO history (list of tuples (it, gbest_cost, gbest_pos))
    if history_pso is not None:
        gens_pso = [x[0] for x in history_pso]
        bests_pso = [x[1] for x in history_pso]
        L = min(len(history_ga["gen"]), len(bests_pso))
        plt.plot(history_ga["gen"][:L], bests_pso[:L], label="PSO best", linestyle=":", linewidth=2, marker='^',
                 markersize=3)

    # TLBO history (list of tuples (it, best_cost, best_pos))
    if history_tlbo is not None:
        gens_tlbo = [x[0] for x in history_tlbo]
        bests_tlbo = [x[1] for x in history_tlbo]
        L = min(len(history_ga["gen"]), len(bests_tlbo))
        plt.plot(history_ga["gen"][:L], bests_tlbo[:L], label="TLBO best", linestyle="-.", linewidth=2, marker='d',
                 markersize=3)

    plt.yscale("log")
    plt.xlabel("Generation / Iteration", fontsize=12)
    plt.ylabel("Cost (lower = better)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
    plt.show()


def plot_cte_comparison(gains_list, labels, dt=0.025):
    """
    Plot CTE time-series for a list of gains.
    Uses simulate_and_cost(Kp,Ki,Kd, dt=...) and shows cost in legend.
    """
    plt.figure(figsize=(10, 6))
    for gains, label in zip(gains_list, labels):
        Kp, Ki, Kd = gains
        cost, cte = simulate_and_cost(Kp, Ki, Kd, dt=dt)
        if len(cte) == 0:
            print(f"[plot_cte] {label} produced empty CTE (likely out-of-bounds).")
            continue
        plt.plot(cte, label=f"{label} (cost {cost:.2f})", alpha=0.8, linewidth=2)
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Cross-track error (m)", fontsize=12)
    plt.title("CTE over time", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()


def save_history_csv(history, filename):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gen", "best_cost", "mean_cost", "best_Kp", "best_Ki", "best_Kd"])
        for i, gen in enumerate(history["gen"]):
            kp, ki, kd = history["best_gains"][i]
            w.writerow([gen, history["best_cost"][i], history["best_cost"][i], kp, ki, kd])
    print(f"[save_history_csv] Written {filename}")


def run_case_study(case_name, path_params=(40, 4, 300), other_cars_fn=None,
                   ga_params=None, sa_params=None, pso_params=None, tlbo_params=None,
                   compare_to_sa=True, compare_to_pso=True, compare_to_tlbo=True,
                   ga_runs=5, sa_runs=5, pso_runs=5, tlbo_runs=5):
    print(f"\n{'=' * 60}")
    print(f"Running case: {case_name}")
    print(f"{'=' * 60}")

    # ---------------------------
    # GA MULTI-RUN
    # ---------------------------
    if ga_params is None:
        ga_params = {}

    print(f"\n[GA] Running {ga_runs} times...")
    t0 = time.time()
    ga_multi = run_ga_multiple(ga_runs, ga_params, other_cars_fn)
    t_ga = time.time() - t0

    print("\n[GA] SUMMARY")
    print(f"  Mean cost: {ga_multi['mean']:.6f}")
    print(f"  Std cost : {ga_multi['std']:.6f}")
    print(f"  Best cost: {min(ga_multi['best_costs']):.6f}")
    print(f"  Best gains: {ga_multi['best_overall']}")
    print(f"  Time: {t_ga:.2f}s")

    # ---------------------------
    # SA MULTI-RUN
    # ---------------------------
    sa_multi = None
    if compare_to_sa:
        print(f"\n[SA] Running {sa_runs} times...")
        t0 = time.time()
        sa_multi = run_sa_multiple(sa_runs, sa_params or {}, other_cars_fn)
        t_sa = time.time() - t0

        print("\n[SA] SUMMARY")
        print(f"  Mean cost: {sa_multi['mean']:.6f}")
        print(f"  Std cost : {sa_multi['std']:.6f}")
        print(f"  Best cost: {min(sa_multi['best_costs']):.6f}")
        print(f"  Best gains: {sa_multi['best_overall']}")
        print(f"  Time: {t_sa:.2f}s")

    # ---------------------------
    # PSO MULTI-RUN
    # ---------------------------
    pso_multi = None
    if compare_to_pso:
        print(f"\n[PSO] Running {pso_runs} times...")
        t0 = time.time()
        pso_multi = run_pso_multiple(pso_runs, pso_params or {}, other_cars_fn)
        t_pso = time.time() - t0

        print("\n[PSO] SUMMARY")
        print(f"  Mean cost: {pso_multi['mean']:.6f}")
        print(f"  Std cost : {pso_multi['std']:.6f}")
        print(f"  Best cost: {min(pso_multi['best_costs']):.6f}")
        print(f"  Best gains: {pso_multi['best_overall']}")
        print(f"  Time: {t_pso:.2f}s")

    # ---------------------------
    # TLBO MULTI-RUN (NEW!)
    # ---------------------------
    tlbo_multi = None
    if compare_to_tlbo:
        print(f"\n[TLBO] Running {tlbo_runs} times...")
        t0 = time.time()
        tlbo_multi = run_tlbo_multiple(tlbo_runs, tlbo_params or {}, other_cars_fn)
        t_tlbo = time.time() - t0

        print("\n[TLBO] SUMMARY")
        print(f"  Mean cost: {tlbo_multi['mean']:.6f}")
        print(f"  Std cost : {tlbo_multi['std']:.6f}")
        print(f"  Best cost: {min(tlbo_multi['best_costs']):.6f}")
        print(f"  Best gains: {tlbo_multi['best_overall']}")
        print(f"  Time: {t_tlbo:.2f}s")

    # ---------------------------
    # CONVERGENCE PLOT
    # ---------------------------
    # Plot using the FIRST run history from each algorithm
    history_ga_raw = ga_multi["histories"][0]
    history_ga = {
        "gen": [x[0] for x in history_ga_raw],
        "best_cost": [x[1] for x in history_ga_raw],
        "mean_cost": [x[1] for x in history_ga_raw],
        "best_gains": [x[2] for x in history_ga_raw],
    }

    history_sa = sa_multi["histories"][0] if sa_multi else None
    history_pso = pso_multi["histories"][0] if pso_multi else None
    history_tlbo = tlbo_multi["histories"][0] if tlbo_multi else None

    plot_convergence(history_ga, history_sa, history_pso, history_tlbo,
                     title=f"Convergence Comparison - {case_name}")

    # ---------------------------
    # CTE COMPARISON
    # ---------------------------
    gains_list = [ga_multi["best_overall"]]
    labels = ["GA best"]

    if sa_multi:
        gains_list.append(sa_multi["best_overall"])
        labels.append("SA best")

    if pso_multi:
        gains_list.append(pso_multi["best_overall"])
        labels.append("PSO best")

    if tlbo_multi:
        gains_list.append(tlbo_multi["best_overall"])
        labels.append("TLBO best")

    plot_cte_comparison(gains_list, labels)

    return {
        "ga_multi": ga_multi,
        "sa_multi": sa_multi,
        "pso_multi": pso_multi,
        "tlbo_multi": tlbo_multi
    }


# ---------------------------
# Multi-run functions
# ---------------------------
def run_ga_multiple(times, ga_params, other_cars_fn=None):
    best_costs = []
    best_gains = []
    histories = []

    for i in range(times):
        print(f"  GA run {i + 1}/{times}", end="\r")
        mapped = dict(
            generations=ga_params.get("generations", 100),
            population_size=ga_params.get("pop_size", 100),
            crossover_alpha=ga_params.get("arith_alpha", 0.5),
            elitism_ratio=ga_params.get("elite_frac", 0.2),
            crossover_ratio=ga_params.get("crossover_frac", 0.6),
            mutation_ratio=ga_params.get("mutation_frac", 0.2),
            mutation_scale=ga_params.get("mutation_scale", (20, 2, 5)),
            visualize_every=None,
            visualize_blocking=False,
            other_cars_fn=other_cars_fn,
            rng_seed=i
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
        print(f"  SA run {i + 1}/{times}", end="\r")
        p = dict(
            initial_gains=(0.5, 0.05, 0.1),
            initial_temp=1.0,
            cooling_rate=0.995,
            iterations=10000,
            step_scale=(20, 2, 5),
            param_bounds=((0.0, 100), (0.0, 10), (0.0, 10)),
            evaluate_fn=simulate_and_cost,
            verbose_every=None,
            visualize_every=None,
            visualize_blocking=False,
            other_cars_fn=other_cars_fn,
            rng_seed=i
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


def run_pso_multiple(times, pso_params, other_cars_fn=None):
    best_costs = []
    best_gains = []
    histories = []

    from PSO import pso_optimize

    for i in range(times):
        print(f"  PSO run {i + 1}/{times}", end="\r")
        mapped = dict(
            iterations=pso_params.get("iterations", 100),
            swarm_size=pso_params.get("swarm_size", 30),
            w=pso_params.get("w", 0.7),
            c1=pso_params.get("c1", 1.5),
            c2=pso_params.get("c2", 1.5),
            bounds=((0.0, 100.0), (0.0, 10.0), (0.0, 10.0)),
            velocity_clip_scale=pso_params.get("velocity_clip_scale", 0.2),
            verbose_every=None,
            visualize_every=None,
            visualize_blocking=False,
            other_cars_fn=other_cars_fn,
            seed=i
        )

        out = pso_optimize(**mapped)
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


def run_tlbo_multiple(times, tlbo_params, other_cars_fn=None):
    """NEW: Multi-run function for TLBO"""
    best_costs = []
    best_gains = []
    histories = []

    from TLBO import tlbo_optimize

    for i in range(times):
        print(f"  TLBO run {i + 1}/{times}", end="\r")
        mapped = dict(
            iterations=tlbo_params.get("iterations", 100),
            population_size=tlbo_params.get("population_size", 30),
            bounds=((0.0, 100.0), (0.0, 10.0), (0.0, 10.0)),
            evaluate_fn=simulate_and_cost,
            verbose_every=None,
            visualize_every=None,
            visualize_blocking=False,
            other_cars_fn=other_cars_fn,
            seed=i
        )

        out = tlbo_optimize(**mapped)
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


# ---------------------------
# Gaussian distribution plotting
# ---------------------------
def plot_gaussian(mean, std, label="Distribution", color=None, shade=True):
    """Plot a Gaussian (normal) PDF from mean and standard deviation."""
    if std == 0:
        x = np.linspace(mean - 1, mean + 1, 400)
        y = np.zeros_like(x)
        center_idx = np.argmin(np.abs(x - mean))
        y[center_idx] = 1.0
        plt.plot(x, y, label=f"{label} (std=0)", linewidth=2, color=color)
        return

    x = np.linspace(mean - 4 * std, mean + 4 * std, 400)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    plt.plot(x, y, label=f"{label} (μ={mean:.3f}, σ={std:.3f})", linewidth=2, color=color)

    if shade:
        plt.fill_between(x, y, alpha=0.2, color=color)


def plot_gaussian_comparison(ga_mean, ga_std, sa_mean=None, sa_std=None,
                             pso_mean=None, pso_std=None, tlbo_mean=None, tlbo_std=None):
    """Plot up to four Gaussian distributions (GA, SA, PSO, TLBO)."""
    plt.figure(figsize=(10, 6))

    plot_gaussian(ga_mean, ga_std, label="GA", color="blue")

    if sa_mean is not None and sa_std is not None:
        plot_gaussian(sa_mean, sa_std, label="SA", color="red")

    if pso_mean is not None and pso_std is not None:
        plot_gaussian(pso_mean, pso_std, label="PSO", color="green")

    if tlbo_mean is not None and tlbo_std is not None:
        plot_gaussian(tlbo_mean, tlbo_std, label="TLBO", color="orange")

    plt.title("Gaussian Distribution of Best Costs Across Runs", fontsize=14)
    plt.xlabel("Cost", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.show()


# ---------------------------
# Default suite with TLBO
# ---------------------------
def default_suite():
    results = {}

    # Case: collision avoidance with traffic
    other_fn = straight_traffic_factory(v=4.5, lane_y=0.0, n=2)
    results["case_collision"] = run_case_study(
        "Collision Avoidance - Traffic Present",
        path_params=(40, 4, 300),
        other_cars_fn=other_fn,
        ga_params={
            "pop_size": 100,
            "generations": 100,
            "elite_frac": 0.20,
            "crossover_frac": 0.60,
            "mutation_frac": 0.20,
            "arith_alpha": 0.5,
            "mutation_scale": (20, 2, 5),
            "tournament_k": 4,
            "rng_seed": 2,
        },
        sa_params={
            "iterations": 10000,
            "initial_temp": 1.0,
            "cooling_rate": 0.995,
        },
        pso_params={
            "iterations": 100,
            "swarm_size": 30,
            "w": 0.7,
            "c1": 1.5,
            "c2": 1.5,
            "velocity_clip_scale": 0.2,
        },
        tlbo_params={
            "iterations": 100,
            "population_size": 30,
        },
        compare_to_sa=True,
        compare_to_pso=True,
        compare_to_tlbo=True,
        ga_runs=5,
        sa_runs=5,
        pso_runs=5,
        tlbo_runs=5,
    )

    return results


# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    tstart = time.time()
    print("\n" + "=" * 60)
    print("RUNNING 4-ALGORITHM OPTIMIZER COMPARISON SUITE")
    print("Algorithms: GA, SA, PSO, TLBO")
    print("=" * 60)

    results = default_suite()

    print("\n" + "=" * 60)
    print(f"ALL CASE STUDIES FINISHED in {time.time() - tstart:.2f} seconds")
    print("=" * 60)

    # Print summary and prepare Gaussian comparison
    for case_name, case_results in results.items():
        print(f"\n{case_name}:")
        print("-" * 60)

        ga = case_results["ga_multi"]
        print(f"  GA   → mean={ga['mean']:.6f}, std={ga['std']:.6f}, best={min(ga['best_costs']):.6f}")
        print(f"         best gains: {ga['best_overall']}")

        if case_results["sa_multi"] is not None:
            sa = case_results["sa_multi"]
            print(f"  SA   → mean={sa['mean']:.6f}, std={sa['std']:.6f}, best={min(sa['best_costs']):.6f}")
            print(f"         best gains: {sa['best_overall']}")

        if case_results["pso_multi"] is not None:
            pso = case_results["pso_multi"]
            print(f"  PSO  → mean={pso['mean']:.6f}, std={pso['std']:.6f}, best={min(pso['best_costs']):.6f}")
            print(f"         best gains: {pso['best_overall']}")

        if case_results["tlbo_multi"] is not None:
            tlbo = case_results["tlbo_multi"]
            print(f"  TLBO → mean={tlbo['mean']:.6f}, std={tlbo['std']:.6f}, best={min(tlbo['best_costs']):.6f}")
            print(f"         best gains: {tlbo['best_overall']}")

    # Generate final Gaussian comparison using first case
    first_case = next(iter(results.values()))

    ga = first_case["ga_multi"]
    ga_mean, ga_std = ga["mean"], ga["std"]

    sa = first_case.get("sa_multi")
    sa_mean, sa_std = (sa["mean"], sa["std"]) if sa else (None, None)

    pso = first_case.get("pso_multi")
    pso_mean, pso_std = (pso["mean"], pso["std"]) if pso else (None, None)

    tlbo = first_case.get("tlbo_multi")
    tlbo_mean, tlbo_std = (tlbo["mean"], tlbo["std"]) if tlbo else (None, None)

    plot_gaussian_comparison(ga_mean, ga_std, sa_mean, sa_std, pso_mean, pso_std, tlbo_mean, tlbo_std)