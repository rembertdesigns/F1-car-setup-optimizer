import joblib
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Load trained ML model
model = joblib.load("models/lap_time_predictor.pkl")

# Define search space
space = [
    Real(0.0, 10.0, name="front_wing_angle"),
    Real(0.0, 15.0, name="rear_wing_angle"),
    Real(3.0, 7.0, name="ride_height"),
    Real(1.0, 10.0, name="suspension_stiffness"),
    Real(50.0, 70.0, name="brake_bias"),
    Real(15.0, 45.0, name="track_temperature"),
    Real(0.8, 1.2, name="grip_level"),
]

def optimize_setup(weights=None):
    if weights is None:
        weights = {
            "lap_time": 1.0,
            "tire_preservation": 0.0,
            "handling_balance": 0.0
        }

    @use_named_args(space)
    def composite_objective(**params):
        X = np.array([[params[k] for k in params]])
        predicted_lap_time = model.predict(X)[0]

        # Tire wear penalty (lower is better)
        tire_penalty = (
            0.2 * (params["suspension_stiffness"] / 10) +
            0.2 * (4 / params["ride_height"]) +  # encourage slightly higher ride height
            0.1 * abs(params["brake_bias"] - 55) / 15
        )

        # Handling imbalance penalty
        imbalance = abs(params["rear_wing_angle"] - params["front_wing_angle"])

        # Composite score to minimize
        score = (
            weights["lap_time"] * predicted_lap_time +
            weights["tire_preservation"] * tire_penalty +
            weights["handling_balance"] * imbalance
        )

        return score

    print("🔍 Running tradeoff optimization...")
    result = gp_minimize(
        func=composite_objective,
        dimensions=space,
        n_calls=40,
        random_state=42
    )

    best_params = dict(zip([dim.name for dim in space], result.x))
    predicted_lap = model.predict(np.array([list(best_params.values())]))[0]

    print("✅ Optimized tradeoff result:")
    for k, v in best_params.items():
        print(f"{k}: {v:.2f}")
    print(f"Predicted Lap Time: {predicted_lap:.2f} sec")

    return best_params, predicted_lap