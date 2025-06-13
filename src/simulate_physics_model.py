import numpy as np
import pandas as pd

def generate_synthetic_data(n_samples=2000, random_seed=42):
    np.random.seed(random_seed)
    data = []

    tire_types = ['soft', 'medium', 'hard', 'intermediate', 'wet']
    weather_types = ['Dry', 'Light Rain', 'Heavy Rain']
    drs_factors = {'Dry': 0.95, 'Light Rain': 1.00, 'Heavy Rain': 1.05}
    grip_map = {'Dry': 1.0, 'Light Rain': 0.85, 'Heavy Rain': 0.65}

    for _ in range(n_samples):
        # Setup parameters
        front_wing = np.random.randint(0, 51)
        rear_wing = np.random.randint(0, 51)
        ride_height = np.random.randint(30, 51)
        suspension = np.random.randint(1, 12)
        brake_bias = np.random.randint(50, 61)

        # Track/environment
        track_temp = np.random.uniform(15, 45)
        grip_level = np.random.uniform(0.8, 1.2)
        weather = np.random.choice(weather_types, p=[0.75, 0.2, 0.05])
        tire = np.random.choice(tire_types)

        # Race context
        lap = np.random.randint(1, 5)
        fuel_weight = np.random.uniform(8, 12)
        traffic = np.random.uniform(0.0, 0.5)
        drs_active = np.random.choice([0, 1], p=[0.3, 0.7])
        safety_car = 0
        vsc = 0

        # Base lap time
        base_lap = 90
        lap_time = base_lap
        lap_time += 0.1 * abs(25 - front_wing)
        lap_time += 0.12 * abs(25 - rear_wing)
        lap_time += 0.2 * abs(40 - ride_height)
        lap_time += 0.15 * abs(6 - suspension)
        lap_time += 0.1 * abs(55 - brake_bias)
        lap_time += 0.1 * abs(30 - track_temp)
        lap_time -= 5 * (grip_level - 1.0)

        # Weather impact
        lap_time *= drs_factors[weather]
        lap_time += (1.0 - grip_map[weather]) * 3.0

        # External race conditions
        lap_time += 0.3 * traffic
        lap_time += 0.2 * fuel_weight
        lap_time += np.random.normal(0, 0.25)

        # Tire type
        tire_map = {
            'soft': -0.3,
            'medium': 0.0,
            'hard': +0.3,
            'intermediate': +0.6,
            'wet': +1.2
        }
        lap_time += tire_map[tire]

        # Add anomaly label (synthetic for ML later)
        is_anomaly = 1 if (ride_height > 48 or abs(front_wing - rear_wing) > 30) else 0

        row = {
            'front_wing_angle': front_wing,
            'rear_wing_angle': rear_wing,
            'ride_height': ride_height,
            'suspension_stiffness': suspension,
            'brake_bias': brake_bias,
            'track_temperature': track_temp,
            'grip_level': grip_level,
            'weather': weather,
            'tire_type': tire,
            'lap': lap,
            'fuel_weight': fuel_weight,
            'traffic': traffic,
            'drs_active': drs_active,
            'safety_car_active': safety_car,
            'vsc_active': vsc,
            'lap_time': lap_time,
            'anomaly': is_anomaly
        }

        data.append(row)

    df = pd.DataFrame(data)

    # One-hot encode tire type + weather
    df = pd.get_dummies(df, columns=["tire_type", "weather"])

    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("data/synthetic_car_setup_v2.csv", index=False)
    print("✅ Enhanced dataset saved to data/synthetic_car_setup_v2.csv")