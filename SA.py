import math
import random
import copy
import matplotlib.pyplot as plt
from PID import PID, Car, elliptical_path




# ==== ADD: road / safety constants & helpers ====
# Road: two lanes (right at y=0, left at y=LANE_W)
LANE_W = 3.6                      # meters
ROAD_Y_MIN = -LANE_W/2            # bottom edge of right lane
ROAD_Y_MAX = +3*LANE_W/2          # top edge of left lane (2 lanes total)

# Car footprint (approx) for safety / collision
CAR_L = 4.5
CAR_W = 1.8
SAFE_GAP_LONG = 10.0              # min bumper-to-bumper gap when same lane (m)
SAFE_LATERAL_MARGIN = 0.3         # extra lateral margin when different lanes (m)

# Speed constraint
V_MAX = 10.0                      # m/s (edit as you like)

# Penalty weights (tune freely)
W_BORDER = 500.0                  # penalize road border violations
W_COLLISION = 1e6                 # big penalty on collision / unsafe gap
W_OVERSPEED = 200.0               # penalize exceeding V_MAX

def same_lane(y1, y2, lane_w=LANE_W):
    """Are both cars in the same lane? (near right-lane center for this model)."""
    return abs(y1 - 0.0) < lane_w/2 and abs(y2 - 0.0) < lane_w/2

def lateral_separation_ok(y1, y2, lane_w=LANE_W):
    """If different lanes, ensure enough lateral separation."""
    return abs(y1 - y2) >= (CAR_W + SAFE_LATERAL_MARGIN)

def border_overshoot(y, y_min=ROAD_Y_MIN, y_max=ROAD_Y_MAX):
    """How far outside the paved road the ego is (0 if inside)."""
    if y < y_min: return (y_min - y)
    if y > y_max: return (y - y_max)
    return 0.0


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
