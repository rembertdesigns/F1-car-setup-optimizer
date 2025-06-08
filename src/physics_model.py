import numpy as np
from scipy.integrate import solve_ivp

# Constants
g = 9.81  # m/s^2
air_density = 1.225  # kg/m^3
mu = 1.4  # typical friction coefficient (rubber on tarmac)

# --- Utility Functions ---
def compute_downforce(speed_kph, downforce_coefficient):
    speed_mps = speed_kph / 3.6
    return 0.5 * air_density * speed_mps**2 * downforce_coefficient

def compute_drag(speed_kph, drag_coefficient):
    speed_mps = speed_kph / 3.6
    return 0.5 * air_density * speed_mps**2 * drag_coefficient

def compute_brake_distance(speed_kph, grip_coefficient=mu):
    speed_mps = speed_kph / 3.6
    return speed_mps**2 / (2 * grip_coefficient * g)

def compute_acceleration(force, mass):
    return force / mass  # a = F / m

# --- Car Parameters from Setup ---
def get_car_params(setup):
    return {
        "mass": 795,  # kg
        "downforce_coeff": 3.0 + (setup["front_wing_angle"] + setup["rear_wing_angle"]) * 0.01,
        "drag_coeff": 1.0 + (setup["front_wing_angle"] + setup["rear_wing_angle"]) * 0.005,
        "mu_brake": 1.8,
        "ride_height_factor": 1.0 - (setup["ride_height"] - 30) * 0.01
    }

# --- Sample Track Layout ---
track = [
    {"type": "straight", "length": 400},
    {"type": "corner", "radius": 80, "angle": 90},
    {"type": "straight", "length": 600},
    {"type": "corner", "radius": 50, "angle": 60},
    {"type": "straight", "length": 1000},
]

# --- Simulations ---
def simulate_straight(length, v0, car):
    def dvdt(t, v):
        drag = 0.5 * air_density * car["drag_coeff"] * v[0]**2
        thrust = 12000  # constant engine force (simplified)
        a = (thrust - drag) / car["mass"]
        return [a]

    def event_distance(t, y):
        return y[0] * t - length

    event_distance.terminal = True
    event_distance.direction = 1

    t_span = (0, 30)
    sol = solve_ivp(dvdt, t_span, [v0], max_step=0.1, events=event_distance)
    v_end = sol.y[0][-1]
    time_taken = sol.t[-1]
    return v_end, time_taken

def simulate_corner(radius, angle_deg, v_in, car):
    angle_rad = np.deg2rad(angle_deg)
    max_lat_accel = car["mu_brake"] * g * car["ride_height_factor"]
    v_max_corner = np.sqrt(max_lat_accel * radius)
    corner_speed = min(v_in, v_max_corner)
    arc_length = radius * angle_rad
    time_taken = arc_length / corner_speed
    return corner_speed, time_taken

def simulate_lap(setup):
    car = get_car_params(setup)
    v = 0
    total_time = 0
    segment_forces = []

    for segment in track:
        if segment["type"] == "straight":
            v_end, t = simulate_straight(segment["length"], v, car)
            drag = compute_drag(v_end * 3.6, car["drag_coeff"])
            downforce = compute_downforce(v_end * 3.6, car["downforce_coeff"])
            segment_forces.append({"type": "straight", "speed_kph": v_end * 3.6, "drag": drag, "downforce": downforce})
            v = v_end
        elif segment["type"] == "corner":
            v_end, t = simulate_corner(segment["radius"], segment["angle"], v, car)
            lat_g = v_end**2 / (segment["radius"] * g)
            segment_forces.append({"type": "corner", "speed_kph": v_end * 3.6, "lat_g": lat_g})
            v = v_end
        total_time += t

    return total_time, segment_forces

# --- Test ---
if __name__ == "__main__":
    test_setup = {
        "front_wing_angle": 25,
        "rear_wing_angle": 25,
        "ride_height": 35,
        "suspension_stiffness": 5,
        "brake_bias": 54
    }
    lap_time, forces = simulate_lap(test_setup)
    print(f"Lap Time: {lap_time:.2f} sec")
    for f in forces:
        print(f)