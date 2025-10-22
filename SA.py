import math
import random
import copy
import matplotlib.pyplot as plt
from PID import PID, Car, elliptical_path


# -------------------------
# Simulation / cost function
# Given PID gains, returns cost (sum of squared CTE)
# You can add more realism or noise if desired
# -------------------------
def simulate_and_cost(Kp, Ki, Kd, verbose=False, dt=0.1, path_params=(40,4,300)):
    # build objects
    pid = PID(Kp, Ki, Kd)
    car = Car(x=0.0, y=0.0, yaw=0.0, velocity=5.0)

    path_x, path_y = elliptical_path(*path_params)

    cte_history = []
    # run along the path points (one desired point per time-step)
    for i in range(len(path_x)):
        desired_x, desired_y = path_x[i], path_y[i]
        cte = desired_y - car.y
        steer = pid.control(cte, dt)
        car.update(steer, dt)
        cte_history.append(cte)

    # cost: sum squared CTE + small penalty on control effort (optional)
    sse = sum(e*e for e in cte_history)
    effort_penalty = 0.0001 * (abs(Kp) + abs(Ki) + abs(Kd))  # small regularizer
    cost = sse + effort_penalty

    if verbose:
        print(f"Sim cost: {cost:.4f}  Kp={Kp:.4f}, Ki={Ki:.6f}, Kd={Kd:.4f}")

    return cost, cte_history


# -------------------------
# Simulated Annealing optimizer
# -------------------------
def simulated_annealing(
    initial_gains=(0.4, 0.002, 0.1),
    initial_temp=1.0,
    cooling_rate=0.995,
    iterations=5000,
    step_scale=(0.2, 0.0005, 0.05),
    param_bounds=((0.0, 5.0), (0.0, 1.0), (0.0, 2.0)),
    evaluate_fn=simulate_and_cost,
    verbose_every=500
):
    random.seed(0)
    current = list(initial_gains)
    current_cost, _ = evaluate_fn(*current)
    best = current[:]
    best_cost = current_cost

    T = initial_temp
    history = []
    accepted = 0

    for it in range(iterations):
        # propose neighbor by adding Gaussian noise scaled by step_scale
        proposal = [
            max(param_bounds[i][0], min(param_bounds[i][1], current[i] + random.gauss(0, step_scale[i])))
            for i in range(3)
        ]

        prop_cost, _ = evaluate_fn(*proposal)

        # acceptance: if better, accept; if worse, accept with exp(-dE/T)
        dE = prop_cost - current_cost
        if dE < 0 or random.random() < math.exp(-dE / T):
            current = proposal
            current_cost = prop_cost
            accepted += 1
            # update best
            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

        # cooling
        T *= cooling_rate
        history.append((it, current_cost, best_cost, T))

        if verbose_every and (it % verbose_every == 0 or it == iterations - 1):
            print(f"it={it:5d} T={T:.4f} cur_cost={current_cost:.4f} best_cost={best_cost:.4f}")

    print(f"SA finished. Accepted moves: {accepted}/{iterations}")
    return {
        "best_gains": tuple(best),
        "best_cost": best_cost,
        "history": history
    }


# -------------------------
# Run SA and visualize results
# -------------------------
if __name__ == "__main__":
    # hyperparameters you can tweak
    init_gains = (0.4, 0.002, 0.1)
    sa_result = simulated_annealing(
        initial_gains=init_gains,
        initial_temp=1.0,
        cooling_rate=0.995,
        iterations=2000,
        step_scale=(0.15, 0.0004, 0.04),
        param_bounds=((0.0, 5.0), (0.0, 0.5), (0.0, 2.0)),
        evaluate_fn=simulate_and_cost,
        verbose_every=200
    )

    print("\nBest gains found:", sa_result["best_gains"], "cost:", sa_result["best_cost"])

    # show performance of best gains
    best_Kp, best_Ki, best_Kd = sa_result["best_gains"]
    cost, cte_history = simulate_and_cost(best_Kp, best_Ki, best_Kd, verbose=True)

    # plot CTE over time
    plt.figure(figsize=(8,4))
    plt.plot(cte_history, label="CTE (best gains)")
    plt.xlabel("Time step")
    plt.ylabel("Cross-track error (m)")
    plt.title("CTE over time with optimized PID gains")
    plt.grid(True)
    plt.legend()
    plt.show()

    # optionally plot best path vs desired
    path_x, path_y = elliptical_path()
    pid = PID(best_Kp, best_Ki, best_Kd)
    car = Car()
    dt = 0.025
    car_x, car_y = [], []
    for i in range(len(path_x)):
        desired_x, desired_y = path_x[i], path_y[i]
        cte = desired_y - car.y
        steer = pid.control(cte, dt)
        car.update(steer, dt)
        car_x.append(car.x)
        car_y.append(car.y)

    plt.figure(figsize=(10,5))
    plt.plot(path_x, path_y, 'r--', label="Desired Path")
    plt.plot(car_x, car_y, 'b-', label="Ego Car (optimized PID)")
    plt.axis('equal')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()
