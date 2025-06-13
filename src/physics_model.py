import numpy as np
from scipy.integrate import solve_ivp

# Constants
g = 9.81  # m/s^2
air_density = 1.225  # kg/m^3
mu = 1.4  # friction (rubber on tarmac)

# Tire temperature model parameters
temp_optimal = 95  # °C
temp_range = (60, 130)  # realistic bounds

def tire_temp_grip_multiplier(temp_c):
    if temp_c < temp_range[0] or temp_c > temp_range[1]:
        return 0.6
    return -0.0005 * (temp_c - temp_optimal)**2 + 1.0  # parabolic curve centered on 95°C

def update_tire_temp(prev_temp, speed_kph, ambient_temp, load_factor):
    delta = 0.05 * (speed_kph / 100) * load_factor
    new_temp = prev_temp + delta - 0.02 * (prev_temp - ambient_temp)  # cools toward ambient
    return np.clip(new_temp, temp_range[0], temp_range[1])

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
    return force / mass

# --- Car Parameters from Setup ---
def get_car_params(setup, fuel_kg=100, weather="Dry"):
    weather_grip_map = {
        "Dry": 1.0,
        "Light Rain": 0.85,
        "Heavy Rain": 0.65
    }
    grip_factor = weather_grip_map.get(weather, 1.0)
    return {
        "mass": 795 + fuel_kg,
        "downforce_coeff": 3.0 + (setup["front_wing_angle"] + setup["rear_wing_angle"]) * 0.01,
        "drag_coeff": 1.0 + (setup["front_wing_angle"] + setup["rear_wing_angle"]) * 0.005,
        "mu_brake": 1.8 * grip_factor,
        "ride_height_factor": 1.0 - (setup["ride_height"] - 30) * 0.01,
        "grip_factor": grip_factor
    }

# --- Sample Track Layout ---
track = [
    {"type": "straight", "length": 400, "drs": True},
    {"type": "corner", "radius": 80, "angle": 90},
    {"type": "straight", "length": 600, "drs": False},
    {"type": "corner", "radius": 50, "angle": 60},
    {"type": "straight", "length": 1000, "drs": True},
]

# --- Simulations ---
def simulate_straight(length, v0, car, drs_active=False):
    drag_coeff = car["drag_coeff"] * (0.7 if drs_active else 1.0)
    def dvdt(t, v):
        drag = 0.5 * air_density * drag_coeff * v[0]**2
        thrust = 12000
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

def simulate_lap(setup, ambient_temp=30.0, fuel_start=100, weather="Dry"):
    v = 0
    total_time = 0
    temp = 85  # initial tire temp
    segment_forces = []
    fuel = fuel_start

    for segment in track:
        car = get_car_params(setup, fuel_kg=fuel, weather=weather)
        if segment["type"] == "straight":
            drs = segment.get("drs", False)
            v_end, t = simulate_straight(segment["length"], v, car, drs_active=drs)
            temp = update_tire_temp(temp, v_end * 3.6, ambient_temp, load_factor=1.0)
            grip_multiplier = tire_temp_grip_multiplier(temp) * car["grip_factor"]
            drag = compute_drag(v_end * 3.6, car["drag_coeff"])
            downforce = compute_downforce(v_end * 3.6, car["downforce_coeff"])
            segment_forces.append({"type": "straight", "speed_kph": v_end * 3.6, "drag": drag, "downforce": downforce, "tire_temp": temp})
            v = v_end
        elif segment["type"] == "corner":
            v_end, t = simulate_corner(segment["radius"], segment["angle"], v, car)
            temp = update_tire_temp(temp, v_end * 3.6, ambient_temp, load_factor=1.5)
            grip_multiplier = tire_temp_grip_multiplier(temp) * car["grip_factor"]
            lat_g = v_end**2 / (segment["radius"] * g)
            segment_forces.append({"type": "corner", "speed_kph": v_end * 3.6, "lat_g": lat_g, "tire_temp": temp})
            v = v_end
        total_time += t
        fuel = max(fuel - 1.7, 0)

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
    lap_time, forces = simulate_lap(setup, ambient_temp=track_conditions["track_temperature"], fuel_start=initial_fuel, weather=weather)
    print(f"Lap Time: {lap_time:.2f} sec")
    for f in forces:
        print(f)