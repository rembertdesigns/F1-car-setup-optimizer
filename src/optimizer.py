import joblib
import json
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

# --- Load Trained Model + Feature List ---
try:
    model = joblib.load("models/lap_time_predictor_v2.pkl")
    with open("models/model_features.json", "r") as f:
        MODEL_FEATURES = json.load(f)
except FileNotFoundError:
    print("❌ Optimizer Error: Trained model or feature list not found.")
    model = None

# --- Search Space: Optimizable Car Setup Parameters ---
space = [
    Integer(0, 50, name="front_wing_angle"),
    Integer(0, 50, name="rear_wing_angle"),
    Integer(30, 50, name="ride_height"),
    Integer(1, 11, name="suspension_stiffness"),
    Integer(50, 60, name="brake_bias")
]

# --- Main Optimization Function ---
def optimize_setup(weights, track_conditions):
    """
    Runs Bayesian Optimization to find the best car setup.

    Args:
        weights (dict): Tradeoff weights (lap_time, tire_preservation, handling_balance)
        track_conditions (dict): Dict with environmental + tire info
    
    Returns:
        tuple: (best_params_dict, predicted_lap_time)
    """
    if model is None:
        raise RuntimeError("Lap time predictor model is not loaded.")

    @use_named_args(space)
    def composite_objective(**setup_params):
        # Initialize all model features
        feature_dict = {f: 0 for f in MODEL_FEATURES}

        # Fixed race context
        feature_dict.update({
            "lap": 2,
            "fuel_weight": 10.0,
            "traffic": 0.0,
            "safety_car_active": 0,
            "vsc_active": 0,
            "track_temperature": track_conditions.get("track_temperature", 30.0),
            "grip_level": track_conditions.get("grip_level", 1.0),
            "rain": track_conditions.get("rain", 0)
        })

        # Optimized car setup
        feature_dict.update(setup_params)

        # One-hot tire compound
        compound = track_conditions.get("tire_compound", "soft")
        tire_key = f"tire_type_{compound}"
        if tire_key in feature_dict:
            feature_dict[tire_key] = 1

        # Predict lap time
        X = pd.DataFrame([feature_dict])[MODEL_FEATURES]
        predicted_lap_time = model.predict(X)[0]

        # Additional penalties
        tire_penalty = (
            0.2 * (setup_params["suspension_stiffness"] / 10.0) +
            0.1 * abs(setup_params["brake_bias"] - 55) / 5.0
        )
        imbalance = abs(setup_params["rear_wing_angle"] - setup_params["front_wing_angle"]) / 50.0

        score = (
            weights["lap_time"] * predicted_lap_time +
            weights["tire_preservation"] * tire_penalty +
            weights["handling_balance"] * imbalance
        )
        return score

    print("🔍 Running Bayesian Optimization...")
    result = gp_minimize(
        func=composite_objective,
        dimensions=space,
        n_calls=50,
        random_state=42,
        n_jobs=-1
    )

    # Retrieve best setup
    best_params_list = result.x
    best_params = {dim.name: val for dim, val in zip(space, best_params_list)}

    # Rebuild full feature vector for final lap time prediction
    best_features_dict = {f: 0 for f in MODEL_FEATURES}
    best_features_dict.update({
        "lap": 2,
        "fuel_weight": 10.0,
        "traffic": 0.0,
        "safety_car_active": 0,
        "vsc_active": 0,
        "track_temperature": track_conditions.get("track_temperature", 30.0),
        "grip_level": track_conditions.get("grip_level", 1.0),
        "rain": track_conditions.get("rain", 0)
    })
    best_features_dict.update(best_params)
    compound = track_conditions.get("tire_compound", "soft")
    tire_key = f"tire_type_{compound}"
    if tire_key in best_features_dict:
        best_features_dict[tire_key] = 1

    best_X = pd.DataFrame([best_features_dict])[MODEL_FEATURES]
    predicted_lap = model.predict(best_X)[0]

    print("✅ Optimized result:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.2f}")
    print(f"Predicted Lap Time: {predicted_lap:.3f} sec")

    return best_params, predicted_lap