import joblib
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# --- Load Trained ML Model ---
try:
    model = joblib.load("models/lap_time_predictor.pkl")
    MODEL_FEATURES = [
        'front_wing_angle', 'rear_wing_angle', 'ride_height',
        'suspension_stiffness', 'brake_bias',
        'track_temperature', 'grip_level'
    ]
except FileNotFoundError:
    print("Optimizer Error: models/lap_time_predictor.pkl not found. The optimizer will not work.")
    model = None

# --- Define Search Space for OPTIMIZABLE setup parameters ---
space = [
    Integer(0, 50, name="front_wing_angle"),
    Integer(0, 50, name="rear_wing_angle"),
    Integer(30, 50, name="ride_height"),
    Integer(1, 11, name="suspension_stiffness"),
    Integer(50, 60, name="brake_bias")
]

def _prepare_feature_vector(setup_params, track_conditions):
    """Create a single-row DataFrame in the correct format for model prediction."""
    feature_dict = {
        "front_wing_angle": setup_params["front_wing_angle"],
        "rear_wing_angle": setup_params["rear_wing_angle"],
        "ride_height": setup_params["ride_height"],
        "suspension_stiffness": setup_params["suspension_stiffness"],
        "brake_bias": setup_params["brake_bias"],
        "track_temperature": track_conditions.get("track_temperature", 30.0),
        "grip_level": track_conditions.get("grip_level", 1.0)
    }
    return pd.DataFrame([feature_dict])[MODEL_FEATURES]

def optimize_setup(weights, track_conditions):
    """Runs single-objective Bayesian Optimization based on weighted tradeoffs."""
    if model is None:
        raise RuntimeError("Lap time predictor model is not loaded. Cannot run optimization.")

    @use_named_args(space)
    def composite_objective(**setup_params):
        # Prepare input features
        X = _prepare_feature_vector(setup_params, track_conditions)
        predicted_lap_time = model.predict(X)[0]

        # Calculate penalties
        tire_penalty = (0.2 * (setup_params["suspension_stiffness"] / 10.0) +
                        0.1 * abs(setup_params["brake_bias"] - 55) / 5.0)
        imbalance = abs(setup_params["rear_wing_angle"] - setup_params["front_wing_angle"]) / 50.0

        # Final score to minimize
        score = (weights["lap_time"] * predicted_lap_time +
                 weights["tire_preservation"] * tire_penalty +
                 weights["handling_balance"] * imbalance)
        return score

    print("🔍 Running Bayesian Optimization...")
    result = gp_minimize(func=composite_objective, dimensions=space, n_calls=50, random_state=42, n_jobs=-1)

    best_params = {dim.name: val for dim, val in zip(space, result.x)}
    best_X = _prepare_feature_vector(best_params, track_conditions)
    predicted_lap = model.predict(best_X)[0]

    print("✅ Optimized tradeoff result found.")
    for k, v in best_params.items():
        print(f"  {k}: {v:.2f}")
    print(f"Predicted Lap Time: {predicted_lap:.3f} sec")

    return best_params, predicted_lap

def run_pareto_optimization(track_conditions, n_steps=15):
    """Runs optimizer multiple times to simulate Pareto front tradeoffs."""
    if model is None:
        raise RuntimeError("Lap time predictor model is not loaded.")

    pareto_front = []
    weight_steps = np.linspace(0, 1, n_steps)

    for lap_time_focus in weight_steps:
        current_weights = {
            "lap_time": lap_time_focus,
            "tire_preservation": 1.0 - lap_time_focus,
            "handling_balance": 0.0
        }

        best_params, _ = optimize_setup(current_weights, track_conditions)
        X = _prepare_feature_vector(best_params, track_conditions)
        final_lap_time = model.predict(X)[0]

        final_tire_penalty = (0.2 * (best_params["suspension_stiffness"] / 10.0) +
                              0.1 * abs(best_params["brake_bias"] - 55) / 5.0)
        final_tire_score = 10 - (final_tire_penalty * 20)

        pareto_front.append({
            "lap_time": final_lap_time,
            "tire_preservation": np.clip(final_tire_score, 0, 10),
            "setup": best_params
        })

    return pd.DataFrame(pareto_front)