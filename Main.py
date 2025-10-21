#initializing problem parameters
num_cars = 500
num_lanes = 6
road_length = 1000  # in meters
# max_speed = 33.33  # in meters per second (120 km/h)
time_step = 1  # in seconds
simulation_duration = 300  # in seconds
#car properties
car_length = 4.5  # in meters
car_width = 1.8  # in meters
# acceleration = 2.5  # in meters per second squared
# deceleration = 7.5  # in meters per second squared
safe_distance = 2.0  # in meters
#car_array with initial positions and speeds
import numpy as np
car_array = np.zeros((num_cars, 4))  # columns: lane, position, speed, acceleration
seed_value = 67
np.random.seed(seed_value)
for i in range(num_cars):
    car_array[i, 0] = np.random.randint(0, num_lanes)  # random lane
    car_array[i, 1] = np.random.uniform(0, road_length)  # random position
    # car_array[i, 2] = np.random.uniform(0, max_speed)  # random speed
    car_array[i, 3] = 0  # initial acceleration
# overtaking Solution [overtake or not,overtake right or left]
solution = np.random.randint(0, 2, size=(num_cars, 2))
print("Initial Car Array:\n", car_array)
print("Overtaking Solution:\n", solution)
#function to calculate conflicts in overtaking maneuvers
def calculate_conflicts(car_array, solution, safe_distance):
    conflicts = 0
    num_cars = car_array.shape[0]
    for i in range(num_cars):
        if solution[i, 0] == 1:  # if car i is overtaking
            target_lane = car_array[i, 0] + (1 if solution[i, 1] == 1 else -1)
            if target_lane < 0 or target_lane >= num_lanes:
                solution[i, 1] = 1 - solution[i, 1]
            for j in range(num_cars):
                if i != j and car_array[j, 0] == target_lane:
                    distance = abs(car_array[i, 1] - car_array[j, 1])
                    if distance < safe_distance:
                        conflicts += 1
    return conflicts
#check if 2 cars are merging into the same lane with the same position
def check_merge_conflicts(car_array, solution):
    merge_conflicts = 0
    num_cars = car_array.shape[0]
    merge_positions = {}
    for i in range(num_cars):
        if solution[i, 0] == 1:  # if car i is overtaking
            target_lane = car_array[i, 0] + (1 if solution[i, 1] == 1 else -1)
            if target_lane < 0 or target_lane >= num_lanes:
                continue
            merge_pos = (target_lane, car_array[i, 1])
            if merge_pos in merge_positions:
                merge_conflicts += 1
            else:
                merge_positions[merge_pos] = i
    return merge_conflicts
#assume if 2 cars are merging into the same lane at the same time, it has a time penalty
def calculate_time_penalty(merge_conflicts, time_penalty_per_conflict=5):
    return merge_conflicts * time_penalty_per_conflict
#measure the distburtion of cars in lanes after overtaking
def lane_distribution(car_array, solution, num_lanes):
    lane_counts = np.zeros(num_lanes)
    num_cars = car_array.shape[0]
    for i in range(num_cars):
        final_lane = car_array[i, 0]
        if solution[i, 0] == 1:  # if car i is overtaking
            final_lane += (1 if solution[i, 1] == 1 else -1)
            if final_lane < 0 or final_lane >= num_lanes:
                final_lane = car_array[i, 0]
        lane_counts[int(final_lane)] += 1
    return lane_counts
#calculate value for lane distribution
def lane_distribution_value(lane_counts):
    mean_count = np.mean(lane_counts)
    variance = np.mean((lane_counts - mean_count) ** 2)
    return variance
#calculate total conflicts and time penalties
conflicts = calculate_conflicts(car_array, solution, safe_distance)
merge_conflicts = check_merge_conflicts(car_array, solution)
time_penalty = calculate_time_penalty(merge_conflicts)
total_cost = conflicts + time_penalty
print("Number of Conflicts:", conflicts)
print("Number of Merge Conflicts:", merge_conflicts)
print("Time Penalty:", time_penalty)
print("Total Cost:", total_cost)
print("Lane Distribution:", lane_distribution(car_array, solution, num_lanes))
print("Lane Distribution Value:", lane_distribution_value(lane_distribution(car_array, solution, num_lanes)))

