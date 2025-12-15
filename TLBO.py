# TLBO.py — Teaching-Learning-Based Optimization for PID tuning
import random
import math
import time
import matplotlib.pyplot as plt
from PID import simulate_and_cost, visualize_pid, straight_traffic_factory


# ====================================================
#   Teaching-Learning-Based Optimization (TLBO)
#   Reference: Rao et al. (2011)
# ====================================================

def tlbo_optimize(
        iterations=100,
        population_size=30,
        bounds=((0.0, 100.0), (0.0, 10.0), (0.0, 10.0)),
        evaluate_fn=simulate_and_cost,
        verbose_every=10,
        visualize_every=None,
        visualize_blocking=True,
        other_cars_fn=None,
        seed=0
):
    """
    Teaching-Learning-Based Optimization (TLBO) for PID gain optimization.

    TLBO is a population-based algorithm inspired by teaching-learning process.
    It has two phases:
    1. Teacher Phase: Learners learn from the best solution (teacher)
    2. Learner Phase: Learners learn from each other through interaction

    Key advantage: NO algorithm-specific parameters to tune!

    Args:
        iterations: Number of generations
        population_size: Number of learners in the class
        bounds: Tuple of (min, max) for each dimension (Kp, Ki, Kd)
        evaluate_fn: Function to evaluate PID gains
        verbose_every: Print progress every N iterations
        visualize_every: Visualize best solution every N iterations
        visualize_blocking: Whether to block during visualization
        other_cars_fn: Optional function for collision testing
        seed: Random seed for reproducibility

    Returns:
        Dictionary with best_gains, best_cost, and history
    """
    random.seed(seed)

    dim = 3  # Kp, Ki, Kd
    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]

    # Initialize population (learners)
    population = [
        [random.uniform(lo[d], hi[d]) for d in range(dim)]
        for _ in range(population_size)
    ]

    # Evaluate initial population
    costs = []
    for learner in population:
        cost, _ = evaluate_fn(*learner, other_cars_fn=other_cars_fn)
        costs.append(cost)

    # Find initial teacher (best learner)
    teacher_idx = min(range(population_size), key=lambda i: costs[i])
    teacher = population[teacher_idx][:]
    teacher_cost = costs[teacher_idx]

    history = []
    history.append((0, teacher_cost, teacher[:]))

    print(f"TLBO start: population={population_size}, iterations={iterations}, seed={seed}")
    print(f"Initial best cost: {teacher_cost:.6f}")

    for iteration in range(1, iterations + 1):

        # ============================================
        # TEACHER PHASE
        # ============================================
        # Calculate mean of each parameter across all learners
        mean = [sum(population[i][d] for i in range(population_size)) / population_size
                for d in range(dim)]

        # Teaching factor (randomly 1 or 2)
        TF = random.choice([1, 2])

        # Each learner learns from teacher
        new_population = []
        new_costs = []

        for i in range(population_size):
            # Create new solution based on teacher
            new_learner = []
            for d in range(dim):
                # Difference between teacher and class mean
                diff = random.random() * (teacher[d] - TF * mean[d])
                new_val = population[i][d] + diff
                # Apply bounds
                new_val = max(lo[d], min(hi[d], new_val))
                new_learner.append(new_val)

            # Evaluate new learner
            new_cost, _ = evaluate_fn(*new_learner, other_cars_fn=other_cars_fn)

            # Accept if better (greedy selection)
            if new_cost < costs[i]:
                new_population.append(new_learner)
                new_costs.append(new_cost)
            else:
                new_population.append(population[i][:])
                new_costs.append(costs[i])

        population = new_population
        costs = new_costs

        # ============================================
        # LEARNER PHASE
        # ============================================
        # Learners interact and learn from each other
        for i in range(population_size):
            # Randomly select another learner
            j = random.randint(0, population_size - 1)
            while j == i:  # Ensure different learner
                j = random.randint(0, population_size - 1)

            # Create new solution based on interaction
            new_learner = []
            for d in range(dim):
                if costs[i] < costs[j]:
                    # Learn from better learner
                    diff = random.random() * (population[i][d] - population[j][d])
                else:
                    # Learn from worse learner (move away)
                    diff = random.random() * (population[j][d] - population[i][d])

                new_val = population[i][d] + diff
                # Apply bounds
                new_val = max(lo[d], min(hi[d], new_val))
                new_learner.append(new_val)

            # Evaluate new learner
            new_cost, _ = evaluate_fn(*new_learner, other_cars_fn=other_cars_fn)

            # Accept if better (greedy selection)
            if new_cost < costs[i]:
                population[i] = new_learner
                costs[i] = new_cost

        # ============================================
        # UPDATE TEACHER (best learner in class)
        # ============================================
        teacher_idx = min(range(population_size), key=lambda i: costs[i])
        if costs[teacher_idx] < teacher_cost:
            teacher = population[teacher_idx][:]
            teacher_cost = costs[teacher_idx]

        # Record history
        history.append((iteration, teacher_cost, teacher[:]))

        # Verbose logging
        if verbose_every and (iteration % verbose_every == 0 or iteration == iterations):
            print(f"Iteration {iteration}/{iterations}: Best cost = {teacher_cost:.6f}, Gains = {teacher}")

        # Visualization checkpoint
        if visualize_every and (iteration % visualize_every == 0):
            print(f"\nVisualizing TLBO best (iteration {iteration})...")
            visualize_pid(*teacher)
            if visualize_blocking:
                time.sleep(0.3)

    # Final visualization
    if visualize_every:
        print(f"\nFinal TLBO visualization...")
        visualize_pid(*teacher)

    print("\nTLBO finished.")
    print(f"Best gains found: {teacher}, Cost: {teacher_cost:.6f}")

    return {
        "best_gains": tuple(teacher),
        "best_cost": teacher_cost,
        "history": history
    }


# ====================================================
#                    TEST SCRIPT
# ====================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing TLBO on PID Tuning Problem")
    print("=" * 60)

    # Test 1: Simple path, no traffic
    print("\n[TEST 1] Simple path, no traffic")
    result1 = tlbo_optimize(
        iterations=100,
        population_size=30,
        bounds=((0.0, 100.0), (0.0, 10.0), (0.0, 10.0)),
        verbose_every=20,
        visualize_every=None,  # Set to 50 to see animations
        other_cars_fn=None,
        seed=42
    )

    # Test 2: With traffic (collision avoidance)
    print("\n[TEST 2] With traffic (collision avoidance)")
    other_cars = straight_traffic_factory(v=4.5, lane_y=0.0, n=2)
    result2 = tlbo_optimize(
        iterations=100,
        population_size=30,
        bounds=((0.0, 100.0), (0.0, 10.0), (0.0, 10.0)),
        verbose_every=20,
        visualize_every=None,
        other_cars_fn=other_cars,
        seed=42
    )

    # Plot convergence comparison
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    history1 = result1["history"]
    iters1 = [h[0] for h in history1]
    costs1 = [h[1] for h in history1]
    plt.plot(iters1, costs1, 'b-', linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title("TLBO Convergence - No Traffic")
    plt.grid(True)
    plt.yscale('log')

    plt.subplot(1, 2, 2)
    history2 = result2["history"]
    iters2 = [h[0] for h in history2]
    costs2 = [h[1] for h in history2]
    plt.plot(iters2, costs2, 'r-', linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title("TLBO Convergence - With Traffic")
    plt.grid(True)
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

    # Final visualization of best solutions
    print("\n[VISUALIZATION] Best solution - No traffic")
    visualize_pid(*result1["best_gains"])

    print("\n[VISUALIZATION] Best solution - With traffic")
    visualize_pid(*result2["best_gains"])