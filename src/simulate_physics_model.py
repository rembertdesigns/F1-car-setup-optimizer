import numpy as np
import pandas as pd

def generate_synthetic_data(n_samples=1000, random_seed=42):
    np.random.seed(random_seed)
    data = []

    for _ in range(n_samples):
        # Random setup values
        front_wing = np.random.uniform(0, 10)
        rear_wing = np.random.uniform(0, 15)
        ride_height = np.random.uniform(3, 7)
        suspension = np.random.uniform(1, 10)
        brake_bias = np.random.uniform(50, 70)

        # Random track conditions
        track_temp = np.random.uniform(15, 45)
        grip_level = np.random.uniform(0.8, 1.2)

        # Simulated lap time
        base_lap = 90
        lap_time = base_lap
        lap_time += 0.1 * abs(6 - front_wing)
        lap_time += 0.15 * abs(10 - rear_wing)
        lap_time += 0.2 * abs(5 - ride_height)
        lap_time += 0.1 * abs(5 - suspension)
        lap_time += 0.05 * abs(60 - brake_bias)
        lap_time += 0.1 * abs(30 - track_temp)  # ideal temp ~30C
        lap_time -= 5 * (grip_level - 1)  # reward extra grip
        lap_time += np.random.normal(0, 0.3)  # noise

        data.append([
            front_wing, rear_wing, ride_height, suspension, brake_bias,
            track_temp, grip_level, lap_time
        ])

    df = pd.DataFrame(data, columns=[
        "front_wing_angle", "rear_wing_angle", "ride_height",
        "suspension_stiffness", "brake_bias",
        "track_temperature", "grip_level", "lap_time"
    ])
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("data/synthetic_car_setup.csv", index=False)
    print("✅ Synthetic data saved to data/synthetic_car_setup.csv")