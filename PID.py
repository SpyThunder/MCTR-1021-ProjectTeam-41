import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import patches  # NEW

CAR_L = 4.5  # car length (m)
CAR_W = 1.8  # car width  (m)

# === constraints (ADD) ===
V_MAX = 10.0                 # m/s cap; overspeed gets penalized
W_OVERSPEED = 200.0          # weight for overspeed penalty
W_COLLISION = 1_000_000.0    # big penalty per collision event


# ===== PID Controller =====
class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def control(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0


# ===== Simple Car Model =====
class Car:
    def __init__(self, x=0, y=0, yaw=0, velocity=5.0):
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

    def reset(self, x=0.0, y=0.0, yaw=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw

def car_corners(x, y, yaw, length=CAR_L, width=CAR_W):
    """Return 4 corner points of the car given center (x, y) and yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    local = [(+hl, +hw), (+hl, -hw), (-hl, -hw), (-hl, +hw)]
    world = [[x + lx*c - ly*s, y + lx*s + ly*c] for lx, ly in local]
    return world

# === collision helpers (ADD under car_corners) ===
def _proj_interval(pts, ax):
    """Project polygon points onto axis ax=(ax_x, ax_y) -> (min,max)."""
    vals = [p[0]*ax[0] + p[1]*ax[1] for p in pts]
    return min(vals), max(vals)

def _sat_overlap(polyA, polyB):
    """Separating Axis Test for two convex quads (our rotated rectangles)."""
    for poly in (polyA, polyB):
        for i in range(4):
            p1 = poly[i]
            p2 = poly[(i+1) % 4]
            edge = (p2[0]-p1[0], p2[1]-p1[1])
            # axis = edge normal (perpendicular)
            ax = (-edge[1], edge[0])
            n = (ax[0]**2 + ax[1]**2) ** 0.5
            if n == 0:
                continue
            ax = (ax[0]/n, ax[1]/n)
            a_min, a_max = _proj_interval(polyA, ax)
            b_min, b_max = _proj_interval(polyB, ax)
            if a_max < b_min or b_max < a_min:
                return False  # separated on this axis
    return True  # all axes overlapped -> collision

# === SAT self-test (ADD just under _sat_overlap) ===
def _sat_self_test():
    # two obvious overlaps (should be True), then a clear gap (False)
    A = [[-1, -1], [-1, +1], [+1, +1], [+1, -1]]           # box at origin
    B = [[0.5, -0.5], [0.5, +0.5], [2.5, +0.5], [2.5, -0.5]]  # overlap on right
    C = [[3.0, -0.5], [3.0, +0.5], [4.0, +0.5], [4.0, -0.5]]  # no overlap
    print("[SAT test] A vs B ->", _sat_overlap(A, B), " (expected True)")
    print("[SAT test] A vs C ->", _sat_overlap(A, C), " (expected False)")

# run once at import (safe, just prints)
_sat_self_test()


# === other cars factory (ADD) ===
def straight_traffic_factory(v=4.5, lane_y=0.0, yaw=0.0, n=2, spacing=25.0):
    """
    Returns a function other_cars_fn(step_idx, t) -> list[(x,y,yaw)] of n cars
    moving straight at speed v, spaced along +x at y=lane_y.
    """
    def other_cars_fn(i, t):
        return [(k*spacing + v*t, lane_y, yaw) for k in range(n)]
    return other_cars_fn


# ===== Define Semi-Elliptical Path =====
def elliptical_path(a=40, b=4, num_points=300):
    xs, ys = [], []
    for t in range(num_points):
        theta = math.pi * t / num_points  # 0 to pi
        x = a * t / num_points
        y = b * math.sin(theta)
        xs.append(x)
        ys.append(y)
    return xs, ys



# ===== Simulation Setup =====
pid = PID(Kp=0.05, Ki=0.0005, Kd=0.15)
car = Car(x=0, y=0, yaw=0, velocity=5.0)
dt = 0.025

path_x, path_y = elliptical_path()
car_x, car_y = [car.x], [car.y]


TARGET_X0 = 6.0      # start this many meters ahead (increase to start farther)
TARGET_V_SCALE = 0.60 # target speed = ego_speed * this factor (slightly slower)
# Target (car being overtaken)
target_x, target_y = TARGET_X0, 0
# --- target car tuning (ADD) ---



# ===== Animation Setup =====
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_aspect('equal', adjustable='box')  # keep 90° angles visually correct
ax.set_xlim(0, max(path_x) + 5)
ax.set_ylim(-6, 6)
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_title("Car Overtaking Maneuver using PID Controller")
ax.grid(True)
ax.plot(path_x, path_y, 'r--', label="Desired Path (Overtake)")
car_dot, = ax.plot([], [], 'bo', markersize=8, label="Ego Car")
target_dot, = ax.plot([], [], 'ko', markersize=8, label="Other Car")
trajectory_line, = ax.plot([], [], 'b-', alpha=0.5)
ego_poly = patches.Polygon([[0, 0]], closed=True, fill=False, ec='b', lw=2, animated=True)
tgt_poly = patches.Polygon([[0, 0]], closed=True, fill=False, ec='k', lw=2, animated=True)
ax.add_patch(ego_poly)
ax.add_patch(tgt_poly)

ax.legend()


def init():
    car_dot.set_data([], [])
    target_dot.set_data([], [])
    trajectory_line.set_data([], [])
    ego_poly.set_xy([[0, 0]])
    tgt_poly.set_xy([[0, 0]])
    return car_dot, target_dot, trajectory_line, ego_poly, tgt_poly


def update(frame):
    global target_x, target_y

    # Move target car straight
    target_x += car.velocity * dt * TARGET_V_SCALE  # slightly slower

    # Find desired point on path
    if frame < len(path_x):
        desired_x, desired_y = path_x[frame], path_y[frame]
    else:
        desired_x, desired_y = path_x[-1], path_y[-1]

    cte = desired_y - car.y  # cross-track error
    steer = pid.control(cte, dt)
    car.update(steer, dt)

    car_x.append(car.x)
    car_y.append(car.y)

    car_dot.set_data([car.x], [car.y])
    target_dot.set_data([target_x], [target_y])
    trajectory_line.set_data(car_x, car_y)
    # Update rectangles (hitboxes)
    ego_poly.set_xy(car_corners(car.x, car.y, car.yaw))
    tgt_poly.set_xy(car_corners(target_x, target_y, 0.0))


    return car_dot, target_dot, trajectory_line, ego_poly, tgt_poly




ani = FuncAnimation(fig, update, frames=len(path_x), init_func=init,
                    blit=True, interval=50, repeat=False)

plt.show()
