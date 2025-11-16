import random
import math
import time
import matplotlib.pyplot as plt
from PID import simulate_and_cost, visualize_pid, straight_traffic_factory
import os
# ====================================================
#                   Genetic Algorithm
# ====================================================

def genetic_algorithm(
    generations,
    population_size,
    crossover_alpha,
    elitism_ratio,
    crossover_ratio,
    mutation_ratio,
    mutation_scale,
    visualize_every,
    visualize_blocking,
    other_cars_fn
):
    random.seed(0)

    # --- Initialize random population ---
    population = [
        [random.uniform(0, 100), random.uniform(0, 10), random.uniform(0, 10)]
        for _ in range(population_size)
    ]

    history = []

    for gen in range(generations):

        # ---- Evaluate population ----
        scored = []
        for ind in population:
            Kp, Ki, Kd = ind
            cost, _ = simulate_and_cost(Kp, Ki, Kd, other_cars_fn=other_cars_fn)
            scored.append((cost, ind))

        scored.sort(key=lambda x: x[0])
        best_cost, best_ind = scored[0]
        history.append((gen, best_cost, best_ind))

        print(f"Generation {gen}: Best cost = {best_cost:.6f} Gains = {best_ind}")
        costs = [simulate_and_cost(*ind, other_cars_fn=other_cars_fn)[0] for ind in population]
        # ---- Visualization ----
        if visualize_every and gen % visualize_every == 0:
            plot_population_3d_enhanced(population, costs,
                                        elite_count=int(elitism_ratio * population_size),
                                        title=f"Generation {gen} population")
            print(f"\nVisualizing best gains (gen {gen})...")
            visualize_pid(*best_ind)
            if visualize_blocking:
                time.sleep(0.3)

        # ---- Elitism ----
        elite_count = int(elitism_ratio * population_size)
        new_pop = [scored[i][1][:] for i in range(elite_count)]

        # ---- Crossover ----
        offspring_count = int(crossover_ratio * population_size)
        for _ in range(offspring_count):
            p1 = random.choice(scored)[1]
            p2 = random.choice(scored)[1]
            child = [
                crossover_alpha * p1[i] + (1 - crossover_alpha) * p2[i]
                for i in range(3)
            ]
            new_pop.append(child)

        # ---- Mutation ----
        mutation_count = int(mutation_ratio * population_size)
        for _ in range(mutation_count):
            parent = random.choice(new_pop)
            child = [
                parent[i] + random.gauss(-mutation_scale[i], mutation_scale[i])
                for i in range(3)
            ]
            new_pop.append(child)

        # ---- Fill or trim population ----
        while len(new_pop) < population_size:
            new_pop.append(random.choice(population))

        population = new_pop[:population_size]

    print("\nGA Finished.")
    print("Best gains found:", best_ind, "Cost:", best_cost)

    return {"best_gains": best_ind, "best_cost": best_cost, "history": history}



# ---------------------------
# Case runner (uses the new GA)
# ---------------------------
def run_case_study(case_name, path_params=(40, 4, 300), other_cars_fn=None,
                   ga_params=None):

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

    # --- Convergence plot ---
    # Convert GA history to required plotting format
    history_ga = {
        "gen": [x[0] for x in ga_result["history"]],
        "best_cost": [x[1] for x in ga_result["history"]],
        "mean_cost": [x[1] for x in ga_result["history"]],  # GA function does not compute mean → reuse best
        "best_gains": [x[2] for x in ga_result["history"]],
    }
    # CTE comparison
    baseline = (0.05, 0.0005, 0.15)
    gains_list = [ga_result["best_gains"]]
    labels = ["Genetic Algorithm best"]

    gains_list.append(baseline)
    labels.append("Baseline")

    return {"ga": ga_result}



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
            "visualize_every": 10,
            "visualize_blocking": True,
        }
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
            "visualize_every": 10,
            "visualize_blocking": True,
        }
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
            "visualize_every": 10,
            "visualize_blocking": True,
        }
    )

    return results

from mpl_toolkits.mplot3d import Axes3D  # required for 3D plots
import numpy as np

def plot_population_3d_enhanced(population, costs, elite_count=0, title="Population distribution in 3D PID space"):
    """
    Plots a 3D scatter of the population with enhanced visualization:
    - Color by cost (fitness)
    - Highlight elites in red
    - Mark best individual with a star

    Args:
        population (list of lists/tuples): Each individual [Kp, Ki, Kd]
        costs (list of floats): Corresponding cost for each individual
        elite_count (int): Number of top individuals considered elite
        title (str): Plot title
    """
    # Extract Kp, Ki, Kd
    Kp = np.array([ind[0] for ind in population])
    Ki = np.array([ind[1] for ind in population])
    Kd = np.array([ind[2] for ind in population])
    costs = np.array(costs)

    # Sort by cost to identify best and elites
    sorted_idx = np.argsort(costs)
    best_idx = sorted_idx[0]

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot non-elite population
    non_elite_idx = sorted_idx[elite_count:]
    sc = ax.scatter(Kp[non_elite_idx], Ki[non_elite_idx], Kd[non_elite_idx],
                    c=costs[non_elite_idx], cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(sc, label="Cost (fitness)")

    # Plot elites in red
    if elite_count > 0:
        elite_idx = sorted_idx[:elite_count]
        ax.scatter(Kp[elite_idx], Ki[elite_idx], Kd[elite_idx],
                   c='red', s=80, label='Elites', edgecolors='k')

    # Highlight the best individual with a star
    ax.scatter(Kp[best_idx], Ki[best_idx], Kd[best_idx],
               c='gold', s=150, marker='*', label='Best', edgecolors='k')

    ax.set_xlabel("Kp")
    ax.set_ylabel("Ki")
    ax.set_zlabel("Kd")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
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
        print(f"{k}: genetic_algorithm best={v['ga']['best_gains']} cost={v['ga']['best_cost']:.6f}")
    print("Results saved in ./ga_results/*.csv")