from __future__ import annotations

import math
import random
import time
from typing import Callable, Dict, List, Tuple, Optional

from PID import simulate_and_cost, visualize_pid

Vector = List[float]
Bounds = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _rand_in_bounds(bounds: Bounds) -> Vector:
    return [random.uniform(b[0], b[1]) for b in bounds]


def _mean_vec(pop: List[Vector]) -> Vector:
    d = len(pop[0])
    out = [0.0] * d
    for p in pop:
        for i in range(d):
            out[i] += p[i]
    n = float(len(pop))
    return [v / n for v in out]


def _levy_flight(beta: float = 1.5) -> float:
    """Scalar Levy step via Mantegna's algorithm."""
    # avoid importing numpy; use random.gauss
    sigma_u = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = random.gauss(0.0, sigma_u)
    v = random.gauss(0.0, 1.0)
    return u / (abs(v) ** (1 / beta) + 1e-12)


def hho_optimize(
    *,
    iterations: int = 500,
    population_size: int = 20,
    bounds: Bounds = ((0.0, 100.0), (0.0, 10.0), (0.0, 10.0)),
    evaluate_fn: Callable[..., Tuple[float, List[float]]] = simulate_and_cost,
    other_cars_fn=None,
    seed: int = 0,
    verbose_every: int = 50,
    visualize_every: Optional[int] = 50,
    visualize_blocking: bool = True,
):
    """Run Harris Hawks Optimizer.

    Returns:
        dict with keys: best_gains (tuple), best_cost (float), history (list)
        history items: (iter, best_cost, best_gains)
    """
    random.seed(seed)

    dim = 3
    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]

    # --- initialize hawks ---
    X: List[Vector] = [_rand_in_bounds(bounds) for _ in range(population_size)]
    F: List[float] = [evaluate_fn(*x, other_cars_fn=other_cars_fn)[0] for x in X]

    best_idx = min(range(population_size), key=lambda i: F[i])
    rabbit = X[best_idx][:]  # best solution
    rabbit_fit = F[best_idx]

    history: List[Tuple[int, float, Tuple[float, float, float]]] = []

    for t in range(iterations):
        Xm = _mean_vec(X)
        for i in range(population_size):
            E0 = 2 * random.random() - 1
            E = 2 * E0 * (1 - (t + 1) / max(1, iterations))  # escaping energy

            q = random.random()
            r = random.random()
            J = 2 * (1 - random.random())  # jump strength

            Xi = X[i]

            # -------------------- Exploration --------------------
            if abs(E) >= 1:
                if q >= 0.5:
                    # perch based on other hawks
                    Xrand = X[random.randrange(population_size)]
                    Xnew = [
                        Xrand[d] - r * abs(Xrand[d] - 2 * r * Xi[d])
                        for d in range(dim)
                    ]
                else:
                    # perch on random tall tree
                    Xnew = [
                        (rabbit[d] - Xm[d]) - r * (lo[d] + random.random() * (hi[d] - lo[d]))
                        for d in range(dim)
                    ]

            # -------------------- Exploitation --------------------
            else:
                # 4 strategies depending on E and r
                if r >= 0.5 and abs(E) >= 0.5:
                    # Soft besiege
                    Xnew = [
                        (rabbit[d] - Xi[d]) - E * abs(J * rabbit[d] - Xi[d])
                        for d in range(dim)
                    ]
                elif r >= 0.5 and abs(E) < 0.5:
                    # Hard besiege
                    Xnew = [
                        rabbit[d] - E * abs(rabbit[d] - Xi[d])
                        for d in range(dim)
                    ]
                elif r < 0.5 and abs(E) >= 0.5:
                    # Soft besiege with progressive rapid dives (Levy)
                    Y = [
                        rabbit[d] - E * abs(J * rabbit[d] - Xi[d])
                        for d in range(dim)
                    ]
                    Z = [
                        Y[d] + random.random() * _levy_flight() * (Y[d] - Xi[d])
                        for d in range(dim)
                    ]
                    # greedy selection between Y and Z
                    Yc = [_clip(Y[d], lo[d], hi[d]) for d in range(dim)]
                    Zc = [_clip(Z[d], lo[d], hi[d]) for d in range(dim)]
                    fy = evaluate_fn(*Yc, other_cars_fn=other_cars_fn)[0]
                    fz = evaluate_fn(*Zc, other_cars_fn=other_cars_fn)[0]
                    Xnew = Y if fy <= fz else Z
                else:
                    # Hard besiege with rapid dives (Levy)
                    Y = [
                        rabbit[d] - E * abs(J * rabbit[d] - Xm[d])
                        for d in range(dim)
                    ]
                    Z = [
                        Y[d] + random.random() * _levy_flight() * (Y[d] - Xm[d])
                        for d in range(dim)
                    ]
                    Yc = [_clip(Y[d], lo[d], hi[d]) for d in range(dim)]
                    Zc = [_clip(Z[d], lo[d], hi[d]) for d in range(dim)]
                    fy = evaluate_fn(*Yc, other_cars_fn=other_cars_fn)[0]
                    fz = evaluate_fn(*Zc, other_cars_fn=other_cars_fn)[0]
                    Xnew = Y if fy <= fz else Z

            # clip to bounds
            Xnew = [_clip(Xnew[d], lo[d], hi[d]) for d in range(dim)]

            fnew, _ = evaluate_fn(*Xnew, other_cars_fn=other_cars_fn)

            # accept if improved
            if fnew < F[i]:
                X[i] = Xnew
                F[i] = fnew

            # update rabbit
            if F[i] < rabbit_fit:
                rabbit_fit = F[i]
                rabbit = X[i][:]

        history.append((t, rabbit_fit, (rabbit[0], rabbit[1], rabbit[2])))

        if verbose_every and (t == 0 or (t + 1) % verbose_every == 0 or t == iterations - 1):
            print(f"HHO it={t+1:5d}  best_cost={rabbit_fit:.6f}  best={rabbit}")

        if visualize_every and (t == 0 or (t + 1) % visualize_every == 0):
            print(f"\nVisualizing HHO best (iter {t+1})...")
            visualize_pid(*rabbit)
            if visualize_blocking:
                time.sleep(0.2)

    return {"best_gains": tuple(rabbit), "best_cost": rabbit_fit, "history": history}


if __name__ == "__main__":
    # Example run (no traffic)
    res = hho_optimize(
        iterations=600,
        population_size=20,
        bounds=((0.0, 100.0), (0.0, 10.0), (0.0, 10.0)),
        seed=0,
        verbose_every=50,
        visualize_every=700,
        other_cars_fn=None,
    )
    print("\nHHO finished.")
    print("Best gains:", res["best_gains"], "Best cost:", res["best_cost"])
    visualize_pid(*res["best_gains"])
