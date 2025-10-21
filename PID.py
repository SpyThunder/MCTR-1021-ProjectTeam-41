import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

# Target (car being overtaken)
target_x, target_y = 0, 0

# ===== Animation Setup =====
fig, ax = plt.subplots(figsize=(10, 5))
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
ax.legend()


def init():
    car_dot.set_data([], [])
    target_dot.set_data([], [])
    trajectory_line.set_data([], [])
    return car_dot, target_dot, trajectory_line


def update(frame):
    global target_x, target_y

    # Move target car straight
    target_x += car.velocity * dt * 0.9  # slightly slower

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

    return car_dot, target_dot, trajectory_line



ani = FuncAnimation(fig, update, frames=len(path_x), init_func=init,
                    blit=True, interval=50, repeat=False)

plt.show()
