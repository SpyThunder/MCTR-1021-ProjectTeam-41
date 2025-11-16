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

    # Map your custom GA parameter names to the imported GA function
    mapped = dict(
        generations       = ga_params.get("generations", 100),
        population_size   = ga_params.get("pop_size", 30),
        crossover_alpha   = ga_params.get("arith_alpha", 0.5),
        elitism_ratio     = ga_params.get("elite_frac", 0.2),
        crossover_ratio   = ga_params.get("crossover_frac", 0.6),
        mutation_ratio    = ga_params.get("mutation_frac", 0.2),
        mutation_scale    = ga_params.get("mutation_scale", (20, 2, 5)),
        visualize_every   = ga_params.get("visualize_every", None),
        visualize_blocking= ga_params.get("visualize_blocking", False),
        other_cars_fn     = other_cars_fn
    )

    # Run the GA
    print("[GA] Running genetic algorithm...")
    t0 = time.time()
    ga_result = genetic_algorithm(**mapped)
    t_ga = time.time() - t0

    print(f"[GA] Done. Best gains: {ga_result['best_gains']}  cost={ga_result['best_cost']:.6f}  time={t_ga:.2f}s")

    # --- SA (optional) ---
    sa_result = None
    if compare_to_sa:
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

        print("[SA] Running simulated annealing...")
        t0 = time.time()
        sa_result = simulated_annealing(**sa_short_params)
        t_sa = time.time() - t0

        print(f"[SA] Done. Best gains: {sa_result['best_gains']} cost={sa_result['best_cost']:.6f} time={t_sa:.2f}s")

    else:
        print("[GA] SA comparison skipped.")

    # --- Convergence plot ---
    # Convert GA history to required plotting format
    history_ga = {
        "gen": [x[0] for x in ga_result["history"]],
        "best_cost": [x[1] for x in ga_result["history"]],
        "mean_cost": [x[1] for x in ga_result["history"]],  # GA function does not compute mean → reuse best
        "best_gains": [x[2] for x in ga_result["history"]],
    }

    history_sa = sa_result["history"] if sa_result else None

    plot_convergence(history_ga, history_sa,
                     title=f"Convergence - {case_name}")

    # CTE comparison
    baseline = (0.05, 0.0005, 0.15)
    gains_list = [ga_result["best_gains"]]
    labels = ["Genetic Algorithm best"]

    if sa_result:
        gains_list.append(sa_result["best_gains"])
        labels.append("SA best")

    gains_list.append(baseline)
    labels.append("Baseline")

    plot_cte_comparison(gains_list, labels, path_params=path_params)

    # Save GA history
    outdir = "ga_results"
    os.makedirs(outdir, exist_ok=True)
    csvname = os.path.join(outdir, f"history_{case_name.replace(' ', '_')}.csv")
    save_history_csv(history_ga, csvname)

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
            "arith_alpha": 0.5,
            "mutation_scale": (0.08, 0.08, 0.08),
            "tournament_k": 3,
            "rng_seed": 0,
        },
        compare_to_sa=True,
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
            "mutation_scale": (0.10, 0.10, 0.10),  # FIXED
            "tournament_k": 3,
            "rng_seed": 1,
        },
        compare_to_sa=True,
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
            "mutation_scale": (0.10, 0.10, 0.10),  # FIXED
            "tournament_k": 4,
            "rng_seed": 2,
        },
        compare_to_sa=True,
    )

    return results



# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    tstart = time.time()
    print("Running genetic_algorithm optimizer suite with fixed percents (this may take a few minutes)...")
    results = default_suite()
    print("\nAll case studies finished in %.2f s" % (time.time() - tstart))
    for k, v in results.items():
        print(f"{k}: genetic_algorithm best={v['ga']['best_gains']} cost={v['ga']['best_cost']:.6f}")
        if v["sa"] is not None:
            print(f"     SA best={v['sa']['best_gains']} cost={v['sa']['best_cost']:.6f}")
    print("Results saved in ./ga_results/*.csv")
