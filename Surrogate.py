"""Surrogate.py — Simple supervised surrogate model for PID cost

Milestone 6 (bonus ML) idea:
  - Use supervised learning to approximate the expensive objective function
    (PID.simulate_and_cost) with a regression surrogate.
  - Then use the surrogate to propose promising PID gains, evaluate the
    best candidates with the real simulator, and iterate.

This is intentionally lightweight and "student-friendly":
  - Polynomial regression (degree=2) fit by least squares (NumPy only)
  - Active learning loop: sample -> fit surrogate -> propose -> validate

Outputs a result dict similar to other optimizers:
  {"best_gains", "best_cost", "history", "model"}
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from PID import simulate_and_cost, visualize_pid

Vector = List[float]
Bounds = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]


def _clip_vec(x: Vector, bounds: Bounds) -> Vector:
    out = []
    for v, (lo, hi) in zip(x, bounds):
        out.append(lo if v < lo else hi if v > hi else v)
    return out


def _rand_in_bounds(bounds: Bounds) -> Vector:
    return [random.uniform(lo, hi) for (lo, hi) in bounds]


def _poly2_features(X: np.ndarray) -> np.ndarray:
    """Build degree-2 polynomial features for 3D inputs.

    For x=[Kp,Ki,Kd] returns:
      [1,
       x1, x2, x3,
       x1^2, x2^2, x3^2,
       x1*x2, x1*x3, x2*x3]
    """
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    feats = np.column_stack(
        [
            np.ones(len(X)),
            x1,
            x2,
            x3,
            x1**2,
            x2**2,
            x3**2,
            x1 * x2,
            x1 * x3,
            x2 * x3,
        ]
    )
    return feats


@dataclass
class Poly2Surrogate:
    """Degree-2 polynomial regression model."""

    w: np.ndarray  # shape (10,)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Phi = _poly2_features(X)
        return Phi @ self.w


def fit_poly2_surrogate(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> Poly2Surrogate:
    """Fit degree-2 polynomial regression with a tiny ridge for stability."""
    Phi = _poly2_features(X)
    # ridge: (Phi^T Phi + λI)w = Phi^T y
    A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    b = Phi.T @ y
    w = np.linalg.solve(A, b)
    return Poly2Surrogate(w=w)


def surrogate_optimize(
    *,
    bounds: Bounds = ((0.0, 100.0), (0.0, 10.0), (0.0, 10.0)),
    evaluate_fn: Callable[..., Tuple[float, List[float]]] = simulate_and_cost,
    other_cars_fn=None,
    seed: int = 0,
    # data + search budget
    initial_samples: int = 60,
    rounds: int = 8,
    proposals_per_round: int = 4000,
    topk_validate: int = 8,
    # optional: visualize
    visualize_every_round: Optional[int] = None,
    visualize_blocking: bool = True,
    verbose: bool = True,
):
    """Surrogate-assisted optimization (supervised ML bonus).

    Process:
      1) Sample N random PID gains in bounds and evaluate true cost.
      2) Fit polynomial regression surrogate.
      3) Sample many candidates, rank by surrogate predicted cost.
      4) Evaluate top-k candidates using true simulator.
      5) Add them to training set and repeat.
    """
    random.seed(seed)
    np.random.seed(seed)

    # --- collect initial dataset ---
    X_data: List[Vector] = []
    y_data: List[float] = []
    for _ in range(initial_samples):
        x = _rand_in_bounds(bounds)
        c, _ = evaluate_fn(*x, other_cars_fn=other_cars_fn)
        X_data.append(x)
        y_data.append(c)

    best_idx = int(np.argmin(y_data))
    best_x = X_data[best_idx][:]
    best_cost = float(y_data[best_idx])

    history: List[Tuple[int, float, Tuple[float, float, float]]] = []
    model: Optional[Poly2Surrogate] = None

    for r in range(rounds):
        X = np.array(X_data, dtype=float)
        y = np.array(y_data, dtype=float)
        model = fit_poly2_surrogate(X, y)

        # --- propose candidates using surrogate ---
        cand = np.array([_rand_in_bounds(bounds) for _ in range(proposals_per_round)], dtype=float)
        yhat = model.predict(cand)
        # take top-k by predicted cost
        idx = np.argsort(yhat)[:topk_validate]
        to_validate = cand[idx]

        # --- validate on the true simulator ---
        round_best = None
        for x in to_validate:
            x_list = _clip_vec(x.tolist(), bounds)
            c, _ = evaluate_fn(*x_list, other_cars_fn=other_cars_fn)
            X_data.append(x_list)
            y_data.append(c)

            if c < best_cost:
                best_cost = float(c)
                best_x = x_list[:]
                round_best = (best_x, best_cost)

        history.append((r, best_cost, (best_x[0], best_x[1], best_x[2])))

        if verbose:
            msg = f"SURR round={r+1:02d}/{rounds}  best_cost={best_cost:.6f}  best={best_x}"
            if round_best is not None:
                msg += "  (improved)"
            print(msg)

        if visualize_every_round and ((r + 1) % visualize_every_round == 0):
            print(f"\nVisualizing Surrogate best (round {r+1})...")
            visualize_pid(*best_x)
            if visualize_blocking:
                time.sleep(0.2)

    return {
        "best_gains": tuple(best_x),
        "best_cost": best_cost,
        "history": history,
        "model": model,
        "dataset_size": len(X_data),
    }


if __name__ == "__main__":
    # quick demo run
    res = surrogate_optimize(
        seed=0,
        initial_samples=60,
        rounds=8,
        proposals_per_round=4000,
        topk_validate=8,
        other_cars_fn=None,
        verbose=True,
        visualize_every_round=None,
    )
    print("\nSurrogate optimization finished.")
    print("Best gains:", res["best_gains"], "Best cost:", res["best_cost"], "Dataset size:", res["dataset_size"])
    visualize_pid(*res["best_gains"])
