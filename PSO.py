# PSO.py — Standard Particle Swarm Optimization for PID tuning
import random
import math
import time
import matplotlib.pyplot as plt
from PID import simulate_and_cost, visualize_pid

# ====================================================
#                     PSO (standard)
# ====================================================
def pso_optimize(
    iterations=1000,
    swarm_size=10,
    w=2,            # inertia weight
    c1=1.5,           # cognitive coeff
    c2=1.5,           # social coeff
    bounds=((0.0, 100.0), (0.0, 10.0), (0.0, 10.0)),
    velocity_clip_scale=0.2,   # v_max = scale * (hi-lo)
    evaluate_fn=simulate_and_cost,
    verbose_every=100,
    visualize_every=500,
    visualize_blocking=True,
    other_cars_fn=None,
    seed=0
):
    """
    Standard PSO to optimize 3 PID gains (Kp, Ki, Kd).

    Returns dict: { "best_gains", "best_cost", "history" }
    where history is list of (iter, best_cost, best_pos)
    """
    random.seed(seed)

    dim = 3
    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    ranges = [hi[i] - lo[i] for i in range(dim)]
    # velocity clamping per-dimension
    vmax = [velocity_clip_scale * r for r in ranges]

    # initialize swarm
    swarm_pos = [
        [random.uniform(lo[d], hi[d]) for d in range(dim)]
        for _ in range(swarm_size)
    ]
    swarm_vel = [
        [random.uniform(-vmax[d], vmax[d]) for d in range(dim)]
        for _ in range(swarm_size)
    ]

    # personal bests
    pbest_pos = [p[:] for p in swarm_pos]
    pbest_cost = []
    for p in pbest_pos:
        cost, _ = evaluate_fn(*p, other_cars_fn=other_cars_fn)
        pbest_cost.append(cost)

    # global best
    gbest_idx = min(range(swarm_size), key=lambda i: pbest_cost[i])
    gbest_pos = pbest_pos[gbest_idx][:]
    gbest_cost = pbest_cost[gbest_idx]

    history = []

    print(f"PSO start: swarm={swarm_size}, iters={iterations}, seed={seed}")
    for it in range(iterations):
        for i in range(swarm_size):
            # velocity update
            for d in range(dim):
                r1 = random.random()
                r2 = random.random()
                cognitive = c1 * r1 * (pbest_pos[i][d] - swarm_pos[i][d])
                social = c2 * r2 * (gbest_pos[d] - swarm_pos[i][d])
                swarm_vel[i][d] = w * swarm_vel[i][d] + cognitive + social
                # clamp velocity
                if swarm_vel[i][d] > vmax[d]:
                    swarm_vel[i][d] = vmax[d]
                if swarm_vel[i][d] < -vmax[d]:
                    swarm_vel[i][d] = -vmax[d]

            # position update
            for d in range(dim):
                swarm_pos[i][d] += swarm_vel[i][d]
                # clamp position to bounds
                if swarm_pos[i][d] < lo[d]:
                    swarm_pos[i][d] = lo[d]
                    swarm_vel[i][d] = 0.0
                if swarm_pos[i][d] > hi[d]:
                    swarm_pos[i][d] = hi[d]
                    swarm_vel[i][d] = 0.0

            # evaluate
            cost, _ = evaluate_fn(*swarm_pos[i], other_cars_fn=other_cars_fn)

            # update personal best
            if cost < pbest_cost[i]:
                pbest_cost[i] = cost
                pbest_pos[i] = swarm_pos[i][:]

                # update global best
                if cost < gbest_cost:
                    gbest_cost = cost
                    gbest_pos = swarm_pos[i][:]

        history.append((it, gbest_cost, gbest_pos[:]))

        # logging
        if verbose_every and (it == 0 or (it + 1) % verbose_every == 0 or it == iterations - 1):
            print(f"it={it+1:5d}  gbest_cost={gbest_cost:.6f}  gbest={gbest_pos}")

        # visualization checkpoint (show best so far)
        if visualize_every and (it == 0 or (it + 1) % visualize_every == 0):
            print(f"\nVisualizing PSO best (iter {it+1})...")
            visualize_pid(*gbest_pos)
            if visualize_blocking:
                time.sleep(0.2)

    print("\nPSO finished.")
    print("Best gains found:", gbest_pos, "Cost:", gbest_cost)
    return {"best_gains": tuple(gbest_pos), "best_cost": gbest_cost, "history": history}


# ====================================================
#                    MAIN SCRIPT
# ====================================================
if __name__ == "__main__":
    # ---- EDIT HERE ONLY (sensible defaults matching GA/SA style) ----
    ITERATIONS = 1000
    SWARM_SIZE = 10
    INERTIA = 1.4
    COGNITIVE = 1.5
    SOCIAL = 1.5
    BOUNDS = ((0.0, 100.0), (0.0, 10.0), (0.0, 10.0))
    VEL_CLIP_SCALE = 0.2
    VIS_EVERY = 100
    VIS_BLOCK = True
    OTHER_CARS = None
    # ----------------------------------------------------------------

# ====================================================
#                 PSO with RING topology
# ====================================================
def pso_ring(
    iterations=1000,
    swarm_size=20,
    w=0.7,
    c1=1.5,
    c2=1.5,
    bounds=((0.0,100.0),(0.0,10.0),(0.0,10.0)),
    velocity_clip_scale=0.2,
    evaluate_fn=simulate_and_cost,
    verbose_every=100,
    visualize_every=500,
    visualize_blocking=True,
    other_cars_fn=None,
    seed=0
):
    random.seed(seed)

    dim = 3
    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    ranges = [hi[i] - lo[i] for i in range(dim)]
    vmax = [velocity_clip_scale * r for r in ranges]

    # swarm initialization (same as PSO)
    swarm_pos = [[random.uniform(lo[d], hi[d]) for d in range(dim)] for _ in range(swarm_size)]
    swarm_vel = [[random.uniform(-vmax[d], vmax[d]) for d in range(dim)] for _ in range(swarm_size)]

    # personal bests
    pbest_pos = [p[:] for p in swarm_pos]
    pbest_cost = [evaluate_fn(*p, other_cars_fn=other_cars_fn)[0] for p in pbest_pos]

    # ring local best for each particle
    def ring_best(i):
        left = (i - 1) % swarm_size
        right = (i + 1) % swarm_size
        candidates = [left, i, right]
        best_idx = min(candidates, key=lambda k: pbest_cost[k])
        return pbest_pos[best_idx]

    history = []

    print(f"PSO (ring) start: swarm={swarm_size}, iters={iterations}, seed={seed}")
    for it in range(iterations):

        for i in range(swarm_size):
            local_best = ring_best(i)

            # velocity update using ring topology
            for d in range(dim):
                r1, r2 = random.random(), random.random()
                cog = c1 * r1 * (pbest_pos[i][d] - swarm_pos[i][d])
                soc = c2 * r2 * (local_best[d] - swarm_pos[i][d])
                swarm_vel[i][d] = w * swarm_vel[i][d] + cog + soc

                # clamp
                swarm_vel[i][d] = max(-vmax[d], min(vmax[d], swarm_vel[i][d]))

            # update position
            for d in range(dim):
                swarm_pos[i][d] += swarm_vel[i][d]
                if swarm_pos[i][d] < lo[d]:
                    swarm_pos[i][d] = lo[d]
                    swarm_vel[i][d] = 0
                if swarm_pos[i][d] > hi[d]:
                    swarm_pos[i][d] = hi[d]
                    swarm_vel[i][d] = 0

            # evaluate
            cost, _ = evaluate_fn(*swarm_pos[i], other_cars_fn=other_cars_fn)

            # update personal best
            if cost < pbest_cost[i]:
                pbest_cost[i] = cost
                pbest_pos[i] = swarm_pos[i][:]

        # record best particle in ring
        g_idx = min(range(swarm_size), key=lambda i: pbest_cost[i])
        g_cost = pbest_cost[g_idx]
        g_pos = pbest_pos[g_idx][:]

        history.append((it, g_cost, g_pos))

        if verbose_every and (it == 0 or (it+1) % verbose_every == 0):
            print(f"it={it+1}  best cost={g_cost:.6f}  best={g_pos}")

        if visualize_every and (it == 0 or (it+1) % visualize_every == 0):
            print(f"\nVisualizing PSO-RING best (iter {it+1})...")
            visualize_pid(*g_pos)
            if visualize_blocking:
                time.sleep(0.2)

    print("\nPSO (ring) finished.")
    print("Best gains found:", g_pos, "Cost:", g_cost)

    return {"best_gains": tuple(g_pos), "best_cost": g_cost, "history": history}

# ====================================================
#               PSO with FOUR CLUSTERS
# ====================================================
def pso_clusters(
    iterations=1000,
    swarm_size=32,      # divisible by 4
    w=0.7,
    c1=1.5,
    c2=1.5,
    bounds=((0.0,100.0),(0.0,10.0),(0.0,10.0)),
    velocity_clip_scale=0.2,
    evaluate_fn=simulate_and_cost,
    verbose_every=100,
    visualize_every=500,
    visualize_blocking=True,
    other_cars_fn=None,
    seed=0
):
    random.seed(seed)

    dim = 3
    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]
    ranges = [hi[i] - lo[i] for i in range(dim)]
    vmax = [velocity_clip_scale * r for r in ranges]

    # swarm initialization
    swarm_pos = [[random.uniform(lo[d], hi[d]) for d in range(dim)] for _ in range(swarm_size)]
    swarm_vel = [[random.uniform(-vmax[d], vmax[d]) for d in range(dim)] for _ in range(swarm_size)]

    pbest_pos = [p[:] for p in swarm_pos]
    pbest_cost = [evaluate_fn(*p, other_cars_fn=other_cars_fn)[0] for p in swarm_pos]

    # divide swarm into 4 equal clusters
    cluster_size = swarm_size // 4
    clusters = [
        list(range(k*cluster_size, (k+1)*cluster_size))
        for k in range(4)
    ]

    def cluster_best(cluster):
        return min(cluster, key=lambda i: pbest_cost[i])

    history = []

    print(f"PSO (clusters) start: swarm={swarm_size}, iters={iterations}, seed={seed}")
    for it in range(iterations):

        # compute best in each cluster
        cluster_bests = [cluster_best(c) for c in clusters]

        for c_idx, cluster in enumerate(clusters):
            best_idx = cluster_bests[c_idx]
            best_pos = pbest_pos[best_idx]

            for i in cluster:

                # update velocity using cluster best only
                for d in range(dim):
                    r1, r2 = random.random(), random.random()
                    cog = c1 * r1 * (pbest_pos[i][d] - swarm_pos[i][d])
                    soc = c2 * r2 * (best_pos[d] - swarm_pos[i][d])
                    swarm_vel[i][d] = w * swarm_vel[i][d] + cog + soc

                    # clamp
                    swarm_vel[i][d] = max(-vmax[d], min(vmax[d], swarm_vel[i][d]))

                # update position
                for d in range(dim):
                    swarm_pos[i][d] += swarm_vel[i][d]
                    if swarm_pos[i][d] < lo[d]:
                        swarm_pos[i][d] = lo[d]
                        swarm_vel[i][d] = 0
                    if swarm_pos[i][d] > hi[d]:
                        swarm_pos[i][d] = hi[d]
                        swarm_vel[i][d] = 0

                # evaluate
                cost, _ = evaluate_fn(*swarm_pos[i], other_cars_fn=other_cars_fn)
                if cost < pbest_cost[i]:
                    pbest_cost[i] = cost
                    pbest_pos[i] = swarm_pos[i][:]

        # global best (optional, for logging)
        g = min(range(swarm_size), key=lambda i: pbest_cost[i])
        g_cost = pbest_cost[g]
        g_pos = pbest_pos[g][:]

        history.append((it, g_cost, g_pos))

        if verbose_every and (it == 0 or (it+1) % verbose_every == 0):
            print(f"it={it+1}  best cost={g_cost:.6f}  best={g_pos}")

        if visualize_every and (it == 0 or (it+1) % visualize_every == 0):
            print(f"\nVisualizing PSO-CLUSTERS best (iter {it+1})...")
            visualize_pid(*g_pos)
            if visualize_blocking:
                time.sleep(0.2)

    print("\nPSO (clusters) finished.")
    print("Best gains found:", g_pos, "Cost:", g_cost)

    return {"best_gains": tuple(g_pos), "best_cost": g_cost, "history": history}

if __name__ == "__main__":

    print("\n=== STAR TOPOLOGY (default) ===")
    result1 = pso_optimize()

    print("\n=== RING TOPOLOGY ===")
    result2 = pso_ring()

    print("\n=== FOUR-CLUSTER TOPOLOGY ===")
    result3 = pso_clusters()