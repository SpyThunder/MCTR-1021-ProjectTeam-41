import math
import random
import time
import matplotlib.pyplot as plt
from PID import (
    PID,
    Car,
    elliptical_path,
    simulate_and_cost,
    visualize_pid,
    straight_traffic_factory,
)

# -------------------------
# Simulated Annealing optimizer (with optional live visualization)
# -------------------------
def simulated_annealing(
    initial_gains=(5.0, 0.5, 1.1),
    initial_temp=1.0,
    cooling_rate=0.995,
    iterations=5000,
    step_scale=(0.2, 0.0005, 0.05),
    param_bounds=((0.0, 5.0), (0.0, 0.5), (0.0, 2.0)),
    evaluate_fn=simulate_and_cost,
    verbose_every=500,
    visualize_every=None,  # e.g. 500 → show progress every 500 iterations
    visualize_blocking=True,
    other_cars_fn=None,    # optional: collision testing
):
    random.seed(0)
    current = list(initial_gains)
    current_cost, _ = evaluate_fn(*current, other_cars_fn=other_cars_fn)
    best = current[:]
    best_cost = current_cost

    T = initial_temp
    history = []
    accepted = 0

    for it in range(iterations):
        # propose new gains
        proposal = [
            max(param_bounds[i][0],
                min(param_bounds[i][1],
                    current[i] + random.gauss(0, step_scale[i])))
            for i in range(3)
        ]

        prop_cost, _ = evaluate_fn(*proposal, other_cars_fn=other_cars_fn)
        dE = prop_cost - current_cost

        # acceptance rule
        if dE < 0 or random.random() < math.exp(-dE / max(T, 1e-12)):
            current = proposal
            current_cost = prop_cost
            accepted += 1
            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

        # cooling
        T *= cooling_rate
        history.append((it, current_cost, best_cost, T))

        # progress log
        if verbose_every and (it == 1 or it % verbose_every == 0 or it == iterations - 1):
            print(f"it={it:5d}  T={T:.6f}  cur_cost={current_cost:.6f}  best_cost={best_cost:.6f}")

        # live visualization checkpoint
        if visualize_every and it > 0 and (it % visualize_every == 0 or it == 1):
            print(f"\n🧠 Visualizing best gains so far (iteration {it})...")
            best_Kp, best_Ki, best_Kd = best
            visualize_pid(best_Kp, best_Ki, best_Kd)
            if visualize_blocking:
                time.sleep(0.2)

    print(f"\nSA finished. Accepted moves: {accepted}/{iterations}")
    return {"best_gains": tuple(best), "best_cost": best_cost, "history": history}


# -------------------------
# Run SA and visualize results
# -------------------------
if __name__ == "__main__":
    # optional: create other cars for collision testing
    # set to None if you don't want collisions
    other_cars_fn = None
    # Example:
    # other_cars_fn = straight_traffic_factory(v=4.5, lane_y=0.0, n=2)

    init_gains = (-5.0, 0.5, 1.1)
    sa_result = simulated_annealing(
        initial_gains=init_gains,
        initial_temp=1.0,
        cooling_rate=0.995,
        iterations=2000,
        step_scale=(0.15, 0.0004, 0.04),
        param_bounds=((-5.0, 5.0), (0.0, 0.5), (0.0, 2.0)),
        evaluate_fn=simulate_and_cost,
        verbose_every=200,
        visualize_every=500,  # show animation every 500 iters
        visualize_blocking=True,
        other_cars_fn=other_cars_fn,
    )

    print("\n🏁 Best gains found:", sa_result["best_gains"], "Cost:", sa_result["best_cost"])

    # === Evaluate best gains ===
    best_Kp, best_Ki, best_Kd = sa_result["best_gains"]
    cost, cte_history = simulate_and_cost(best_Kp, best_Ki, best_Kd, verbose=True)

    # Plot CTE over time
    plt.figure(figsize=(8, 4))
    plt.plot(cte_history, label="CTE (best gains)")
    plt.xlabel("Time step")
    plt.ylabel("Cross-track error (m)")
    plt.title("CTE over time with optimized PID gains")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Visualize the optimized PID in action 🚗💨
    visualize_pid(best_Kp, best_Ki, best_Kd)
