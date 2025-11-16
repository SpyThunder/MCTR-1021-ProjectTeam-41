import random
import math
import time
import matplotlib.pyplot as plt
from PID import simulate_and_cost, visualize_pid

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

        # ---- Visualization ----
        if visualize_every and gen % visualize_every == 0:
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


# ====================================================
#                    MAIN SCRIPT
# ====================================================

if __name__ == "__main__":

    # ---- EDIT HERE ONLY ----
    GENERATIONS = 1000
    POP_SIZE = 10
    ALPHA = 0.5               # crossover weight
    ELITISM = 0.2
    CROSSOVER = 0.6
    MUTATION = 0.2
    MUT_SCALE = (20, 2, 5)
    VIS_EVERY = 100
    BLOCK = True
    OTHER_CARS = None
    # ------------------------

    # ---- Run GA ----
    result = genetic_algorithm(
        generations=GENERATIONS,
        population_size=POP_SIZE,
        crossover_alpha=ALPHA,
        elitism_ratio=ELITISM,
        crossover_ratio=CROSSOVER,
        mutation_ratio=MUTATION,
        mutation_scale=MUT_SCALE,
        visualize_every=VIS_EVERY,
        visualize_blocking=BLOCK,
        other_cars_fn=OTHER_CARS
    )

    # ---- Evaluate best solution ----
    Kp, Ki, Kd = result["best_gains"]
    cost, cte_history = simulate_and_cost(Kp, Ki, Kd, verbose=True)

    # ---- Plot CTE history ----
    plt.figure(figsize=(8, 4))
    plt.plot(cte_history)
    plt.xlabel("Time step")
    plt.ylabel("CTE (m)")
    plt.title("CTE over time (Best GA Gains)")
    plt.grid(True)
    plt.show()

    # ---- Visualize PID motion ----
    visualize_pid(Kp, Ki, Kd)
