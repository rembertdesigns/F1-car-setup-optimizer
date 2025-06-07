import joblib
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# --- Load Trained ML Model ---
# This model expects a 14-feature input.
try:
    model = joblib.load("models/lap_time_predictor.pkl")
    # This is the exact list of features the model was trained on
    MODEL_FEATURES = [
        'lap', 'tire_wear', 'traffic', 'fuel_weight', 'track_temperature', 
        'grip_factor', 'rain', 'safety_car_active', 'vsc_active', 
        'tire_type_soft', 'tire_type_medium', 'tire_type_hard', 
        'tire_type_intermediate', 'tire_type_wet'
    ]
except FileNotFoundError:
    print("Optimizer Error: models/lap_time_predictor.pkl not found. The optimizer will not work.")
    model = None

# --- Define Search Space for OPTIMIZABLE parameters ---
# These are the parameters the user can change with sliders.
# Note: We've removed track_temperature and grip_level as they are fixed conditions, not setup choices.
space = [
    Integer(0, 50, name="front_wing_angle"),
    Integer(0, 50, name="rear_wing_angle"),
    Integer(30, 50, name="ride_height"),
    Integer(1, 11, name="suspension_stiffness"),
    Integer(50, 60, name="brake_bias")
]

def optimize_setup(weights, track_conditions):
    """
    Runs Bayesian Optimization to find the best car setup.

    Args:
        weights (dict): A dictionary with weights for 'lap_time', 'tire_preservation', 'handling_balance'.
        track_conditions (dict): A dictionary with fixed environmental parameters for the simulation.
                                 e.g., {'track_temperature': 30.0, 'grip_level': 1.0, 'tire_compound': 'soft'}
    
    Returns:
        tuple: A tuple containing (best_params_dict, predicted_lap_time)
    """
    if model is None:
        raise RuntimeError("Lap time predictor model is not loaded. Cannot run optimization.")

    # The decorator that converts the list of parameters from the optimizer into keyword arguments
    @use_named_args(space)
    def composite_objective(**setup_params):
        """
        This is the function the optimizer will try to minimize.
        It takes the setup parameters being tested and returns a single score.
        """
        
        # --- Prepare the full 14-feature vector for the ML model ---
        # Start with a dictionary of all features, initialized to 0
        feature_dict = {feature: 0 for feature in MODEL_FEATURES}

        # 1. Update with fixed track conditions
        feature_dict.update({
            "track_temperature": track_conditions.get('track_temperature', 30.0),
            "grip_factor": track_conditions.get('grip_level', 1.0),
            # Set simulation constants for a single hot-lap prediction
            "lap": 2, # Assume it's an early lap
            "tire_wear": 5.0, # Fresh tires
            "traffic": 0.0, # No traffic for a qualifying-style lap
            "fuel_weight": 10.0, # Low fuel
            "rain": 0, "safety_car_active": 0, "vsc_active": 0
        })

        # 2. Update with the current setup parameters being tested by the optimizer
        feature_dict.update(setup_params)
        
        # 3. One-hot encode the assumed tire compound for this setup
        # For a setup optimizer, we often assume the fastest tire (soft) is used.
        assumed_tire = track_conditions.get('tire_compound', 'soft')
        feature_dict[f'tire_type_{assumed_tire}'] = 1

        # Create a single-row DataFrame in the correct order for prediction
        X = pd.DataFrame([feature_dict])[MODEL_FEATURES]

        # Predict the raw lap time from the model
        predicted_lap_time = model.predict(X)[0]

        # --- Calculate Penalty Scores based on the setup ---
        
        # Tire wear penalty (lower is better). Based on setup aggressiveness.
        tire_penalty = (
            0.2 * (setup_params["suspension_stiffness"] / 10.0) + # Stiffer suspension = more wear
            0.1 * abs(setup_params["brake_bias"] - 55) / 5.0    # Extreme brake bias can affect wear
        )

        # Handling imbalance penalty (aero balance)
        imbalance = abs(setup_params["rear_wing_angle"] - setup_params["front_wing_angle"]) / 50.0

        # --- Calculate the final composite score to minimize ---
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
        n_calls=50,  # Number of different setups to test
        random_state=42,
        n_jobs=-1 # Use all available CPU cores
    )

    # Extract the best setup parameters found
    best_params_list = result.x
    best_params = {dim.name: val for dim, val in zip(space, best_params_list)}
    
    # To get the final predicted lap time, we need to prepare the features for the best setup again
    final_features_df = pd.DataFrame([best_params])
    # ... (Need to add the rest of the 14 features as done in composite_objective) ...
    # For simplicity, we can just re-run the objective function with the best params and only get the lap time part
    # Or, we can extract it from the optimization result if the objective returned it.
    # The 'result.fun' attribute holds the best *score*, not the lap time.
    
    # Re-predict just the lap time for the best found setup
    best_features_dict = {feature: 0 for feature in MODEL_FEATURES}
    best_features_dict.update(track_conditions)
    best_features_dict.update(best_params)
    best_features_dict[f'tire_type_{track_conditions.get("tire_compound", "soft")}'] = 1
    best_X = pd.DataFrame([best_features_dict])[MODEL_FEATURES]
    predicted_lap = model.predict(best_X)[0]


    print("✅ Optimized tradeoff result found:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.2f}")
    print(f"Predicted Lap Time for this setup: {predicted_lap:.3f} sec")

    return best_params, predicted_lap