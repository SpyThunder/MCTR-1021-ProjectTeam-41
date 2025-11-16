# PID.py — cleaned & fixed
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import patches

# ---- Constants ----
CAR_L = 4.5  # car length (m)
CAR_W = 1.8  # car width  (m)

# === constraints (ADD) ===
V_MAX = 10.0                 # m/s cap; overspeed gets penalized
W_OVERSPEED = 200.0          # weight for overspeed penalty
W_COLLISION = 10.0    # big penalty per collision event

target_x0 = 6.0  # initial x position of the other car
tv_scale = 0.5    # other car speed scale (fraction of ego speed)

# --- road bounds (ADD) ---
LANE_W = 3.6
# ROAD_Y_MIN = -LANE_W/2          # bottom edge of right lane
# ROAD_Y_MAX = +3*LANE_W/2        # top edge of left lane (2 lanes total)

ROAD_Y_MIN = -100          # bottom edge of right lane
ROAD_Y_MAX = 100

# optional cost kick when leaving the road
W_BORDER = 1000.0               # weight for road-exit penalty


# ===== PID Controller =====
class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0.0
        self.integral = 0.0

    def control(self, error, dt):
        # guard dt > 0
        if dt <= 0:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / dt
        self.integral += error * dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0


# ===== Simple Car Model =====
class Car:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, velocity=5.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.velocity = velocity
        self.L = 2.5  # wheelbase

    def update(self, delta, dt):
        delta = max(-math.radians(30), min(math.radians(30), delta))  # steering limit
        self.x += self.velocity * math.cos(self.yaw) * dt
        self.y += self.velocity * math.sin(self.yaw) * dt
        self.yaw += (self.velocity / self.L) * math.tan(delta) * dt

    def reset(self, x=0.0, y=0.0, yaw=0.0, velocity=None):
        self.x = x
        self.y = y
        self.yaw = yaw
        if velocity is not None:
            self.velocity = velocity


# ---- geometry helpers for rotated rectangle (car) ----
def car_corners(x, y, yaw, length=CAR_L, width=CAR_W):
    """Return list of 4 corner points [[x,y],...] of a rectangle centered at (x,y) rotated by yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    local = [(+hl, +hw), (+hl, -hw), (-hl, -hw), (-hl, +hw)]
    world = [[x + lx * c - ly * s, y + lx * s + ly * c] for lx, ly in local]
    return world


# === collision helpers ===
def _proj_interval(pts, ax):
    """Project polygon points onto axis ax=(ax_x, ax_y) -> (min,max)."""
    vals = [p[0] * ax[0] + p[1] * ax[1] for p in pts]
    return min(vals), max(vals)


def _sat_overlap(polyA, polyB):
    """Separating Axis Test for two convex quads (rotated rectangles)."""
    for poly in (polyA, polyB):
        for i in range(4):
            p1 = poly[i]
            p2 = poly[(i + 1) % 4]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            # axis = edge normal (perpendicular)
            ax = (-edge[1], edge[0])
            n = math.hypot(ax[0], ax[1])
            if n == 0:
                continue
            ax = (ax[0] / n, ax[1] / n)
            a_min, a_max = _proj_interval(polyA, ax)
            b_min, b_max = _proj_interval(polyB, ax)
            if a_max < b_min or b_max < a_min:
                return False  # separated on this axis
    return True  # all axes overlapped -> collision

# --- border overshoot helper (ADD) ---
def border_overshoot(y, y_min=ROAD_Y_MIN, y_max=ROAD_Y_MAX):
    if y < y_min: return (y_min - y)
    if y > y_max: return (y - y_max)
    return 0.0



# === SAT self-test (optional, safe to run at import) ===
def _sat_self_test():
    A = [[-1, -1], [-1, +1], [+1, +1], [+1, -1]]           # box at origin
    B = [[0.5, -0.5], [0.5, +0.5], [2.5, +0.5], [2.5, -0.5]]  # overlap on right
    C = [[3.0, -0.5], [3.0, +0.5], [4.0, +0.5], [4.0, -0.5]]  # no overlap
    print("[SAT test] A vs B ->", _sat_overlap(A, B), " (expected True)")
    print("[SAT test] A vs C ->", _sat_overlap(A, C), " (expected False)")


# run quick test once at import (it's harmless)
_sat_self_test()


# === other cars factory (returns a function that gives other-car poses at time t) ===
def straight_traffic_factory(v=4.5, lane_y=0.0, yaw=0.0, n=2, spacing=25.0):
    """
    Returns other_cars_fn(t) -> list of tuples (x,y,yaw) for n cars
    moving straight at speed v, spaced along +x at y=lane_y.
    """
    def other_cars_fn(t):
        return [(k * spacing + v * t, lane_y, yaw) for k in range(n)]
    return other_cars_fn


# ===== Define Semi-Elliptical Path =====
def elliptical_path():
    """
    Simple semi-elliptical overtaking path.
    Path type 1.
    """
    a = 40.0          # total x length (m)
    b = 4.0           # peak lateral offset (m)
    num_points = 300  # path resolution

    xs, ys = [], []
    for t in range(num_points):
        theta = math.pi * t / (num_points - 1)  # 0 .. pi
        x = a * t / (num_points - 1)
        y = b * math.sin(theta)
        xs.append(x)
        ys.append(y)
    return xs, ys


def medium_path():
    """
    Medium-complexity S-curve style path.
    Path type 2.
    """
    a = 40.0          # total x length
    b = 3.0           # lateral amplitude
    num_points = 400  # a bit denser

    xs, ys = [], []
    for t in range(num_points):
        s = t / (num_points - 1)   # goes 0..1
        x = a * s
        # smooth S-shape: start in lane, change, then flatten
        y = b * math.sin(math.pi * (s - 0.5))
        xs.append(x)
        ys.append(y)
    return xs, ys


def complex_path():
    """
    Higher-complexity path with multiple bends.
    Path type 3.
    """
    a = 40.0
    b = 3.5
    num_points = 500

    xs, ys = [], []
    for t in range(num_points):
        s = t / (num_points - 1)  # 0..1
        x = a * s
        base = b * math.sin(2.0 * math.pi * s)      # main oscillation
        wiggle = 0.7 * math.sin(6.0 * math.pi * s)  # extra complexity
        y = base + wiggle
        xs.append(x)
        ys.append(y)
    return xs, ys


def generate_path(path_type=1):
    """
    Select one of the predefined paths by integer id.

    path_type:
      1 -> elliptical_path  (simple overtaking)
      2 -> medium_path      (S-curve)
      3 -> complex_path     (wiggly / complex)
    """
    if path_type == 1:
        return elliptical_path()
    elif path_type == 2:
        return medium_path()
    elif path_type == 3:
        return complex_path()
    else:
        # default fallback
        return elliptical_path()


# ===== Simulation / cost function =====
def simulate_and_cost(
    Kp, Ki, Kd,
    *,
    dt=0.025,
    path_type=3,
    bounds=((0.0, 100), (0.0, 10), (0.0, 10)),
    other_cars_fn=None,
    verbose=False
):
    """
    Simulate a PID-controlled car following a semi-elliptical path.

    Includes:
      - Hard gain bounds (skip evaluation if out of range)
      - Overspeed penalty (if velocity > V_MAX)
      - Collision penalty (if overlapping with other cars from other_cars_fn)

    Returns:
        cost (float): total cost value (lower is better)
        cte_history (list): cross-track error over time
    """
    # 0) Hard gain bounds
    (kp_lo, kp_hi), (ki_lo, ki_hi), (kd_lo, kd_hi) = bounds
    if not (kp_lo <= Kp <= kp_hi and ki_lo <= Ki <= ki_hi and kd_lo <= Kd <= kd_hi):
        return 1e12, []  # huge penalty for out-of-bounds gains

        # 1) Setup simulation
    pid = PID(Kp, Ki, Kd)

    # get path from selector
    path_x, path_y = generate_path(path_type)
    N = len(path_x)
    total_dx = path_x[-1] - path_x[0] if N > 1 else 0.0

    # choose velocity so we roughly reach the end of the path in N steps
    v = 0.0 if N <= 1 else (0.5 + total_dx) / (dt * (N - 1))

    car = Car(x=0.0, y=0.0, yaw=0.0, velocity=v)

    cte_history = []
    collisions = 0
    overspeed_accum = 0.0


    # === Target setup (same as visualization) ===
    target_x = target_x0
    target_y = 0.0
    target_yaw = 0.0
    target_v_scale = tv_scale

    steps_used = 0            # ADD
    left_road = False         # ADD
    overshoot_amt = 0.0       # ADD


    # 2) Simulation loop
    for i in range(len(path_x)):
        t = i * dt
        desired_y = path_y[i]
        cte = desired_y - car.y
        steer = pid.control(cte, dt)
        car.update(steer, dt)
        cte_history.append(cte)
                # --- early stop if car exits road (ADD) ---
        if not (ROAD_Y_MIN <= car.y <= ROAD_Y_MAX):
            left_road = True
            overshoot_amt = border_overshoot(car.y)  # how far outside (meters)
            steps_used += 1                           # count this last step
            break

        # === Update target and check collision with ego ===
        target_x += car.velocity * dt * target_v_scale  # target moves forward
        # === Compute polygons with identical parameters to visualization ===
        ego_poly = car_corners(
            car.x,
            car.y,
            car.yaw,
            length=CAR_L,
            width=CAR_W,
        )
        target_poly = car_corners(
            target_x,
            target_y,
            target_yaw,
            length=CAR_L,
            width=CAR_W,
        )

        # === Apply small tolerance to avoid floating-point edge contact ===
        overlap = _sat_overlap(ego_poly, target_poly)
        if overlap:
            # Check centroid distance to filter out micro-overlaps (< 5 cm)
            dx = car.x - target_x
            dy = car.y - target_y
            if math.hypot(dx, dy) < 0.95 * CAR_L:
                collisions += 1

        # Overspeed penalty
        if car.velocity > V_MAX:
            overspeed_accum += (car.velocity - V_MAX) ** 2

        # Collision penalty
        if other_cars_fn is not None:
            try:
                others = other_cars_fn(t)
            except TypeError:
                others = other_cars_fn(i, t)

            ego_poly = car_corners(car.x, car.y, car.yaw)
            for ox, oy, oyaw in others:
                other_poly = car_corners(ox, oy, oyaw)
                if _sat_overlap(ego_poly, other_poly):
                    collisions += 1
        steps_used += 1

    # 3) Compute cost
        # time term if you’re using it; safe even if you’re not
    total_time = steps_used * dt  # ADD (optional)

    # --- border penalty (ADD) ---
    border_penalty = W_BORDER * (1.0 + overshoot_amt) if left_road else 0.0

    sse = sum(e * e for e in cte_history)
    effort_penalty = 0.0001 * (abs(Kp) + abs(Ki) + abs(Kd))
    cost = (
        sse
        + effort_penalty
        + W_OVERSPEED * overspeed_accum
        + W_COLLISION * collisions
        + border_penalty  
    )

    if verbose:
        print(
            f"Cost={cost:.4f}  SSE={sse:.4f}  Collisions={collisions} "
            f"Overspeed={overspeed_accum:.2f}  LeftRoad={left_road}  "
            f"Gains=({Kp:.4f},{Ki:.6f},{Kd:.4f})"
        )


    return cost, cte_history



# ===== Visualization function (safe to import) =====
def visualize_pid(Kp=0.05, Ki=0.0005, Kd=0.15, dt=0.025,
                  path_type=3,
                  show_hitboxes=True):

    """
    Animate a single run of the PID controller following the semi-elliptical path.
    Call visualize_pid(...) from other modules (e.g. SA.py) with optimized gains.
    """
    pid = PID(Kp, Ki, Kd)

    # get path and set speed to match its length
    path_x, path_y = generate_path(path_type)
    N = len(path_x)
    total_dx = path_x[-1] - path_x[0] if N > 1 else 0.0
    v = 0.0 if N <= 1 else (0.5 + total_dx) / (dt * (N - 1))

    car = Car(x=0.0, y=0.0, yaw=0.0, velocity=v)
    car_x, car_y = [car.x], [car.y]


    target_v_scale = tv_scale
    target_x = target_x0
    target_y = 0.0

    # setup figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, max(path_x) + 5)
    ax.set_ylim(-6, 6)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Car Overtaking Maneuver using PID Controller")
    ax.grid(True)
    ax.plot(path_x, path_y, 'r--', label="Desired Path (Overtake)")

    car_dot, = ax.plot([], [], 'bo', markersize=6, label="Ego Car")
    target_dot, = ax.plot([], [], 'ko', markersize=6, label="Other Car")
    trajectory_line, = ax.plot([], [], 'b-', alpha=0.6, label="Ego Trajectory")

    ego_poly = patches.Polygon([[0, 0]], closed=True, fill=False, ec='b', lw=2, animated=True) if show_hitboxes else None
    tgt_poly = patches.Polygon([[0, 0]], closed=True, fill=False, ec='k', lw=2, animated=True) if show_hitboxes else None
    if ego_poly is not None:
        ax.add_patch(ego_poly)
    if tgt_poly is not None:
        ax.add_patch(tgt_poly)

    ax.legend()

    def init():
        car_dot.set_data([], [])
        target_dot.set_data([], [])
        trajectory_line.set_data([], [])
        if ego_poly is not None:
            ego_poly.set_xy([[0, 0]])
        if tgt_poly is not None:
            tgt_poly.set_xy([[0, 0]])
        return (car_dot, target_dot, trajectory_line, ego_poly, tgt_poly) if ego_poly is not None else (car_dot, target_dot, trajectory_line)

    def update(frame):
        nonlocal target_x, target_y
        # advance target
        target_x += car.velocity * dt * target_v_scale

        # desired point
        if frame < len(path_x):
            desired_y = path_y[frame]
        else:
            desired_y = path_y[-1]

        cte = desired_y - car.y
        steer = pid.control(cte, dt)
        car.update(steer, dt)

        car_x.append(car.x)
        car_y.append(car.y)

                # --- stop the animation if we exit the road (ADD) ---
        if not (ROAD_Y_MIN <= car.y <= ROAD_Y_MAX):
            print(f"Exited road at frame {frame}: y={car.y:.2f}. Stopping.")
            plt.close(fig)  # closes the figure; ends the animation
            # return artists so blit has something valid
            return (car_dot, target_dot, trajectory_line, ego_poly, tgt_poly) if ego_poly is not None else (car_dot, target_dot, trajectory_line)


        car_dot.set_data([car.x], [car.y])
        target_dot.set_data([target_x], [target_y])
        trajectory_line.set_data(car_x, car_y)

        if ego_poly is not None:
            ego_poly.set_xy(car_corners(car.x, car.y, car.yaw))
        if tgt_poly is not None:
            tgt_poly.set_xy(car_corners(target_x, target_y, 0.0))

        return (car_dot, target_dot, trajectory_line, ego_poly, tgt_poly) if ego_poly is not None else (car_dot, target_dot, trajectory_line)

    ani = FuncAnimation(fig, update, frames=len(path_x), init_func=init,
                        blit=True, interval=50, repeat=False)
    plt.show()


# Run visualization if executed directly
if __name__ == "__main__":
    visualize_pid()


