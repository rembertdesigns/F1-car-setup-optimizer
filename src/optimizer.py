import joblib
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Load trained model
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

@use_named_args(space)
def objective(**params):
    X = np.array([[params[key] for key in params]])
    pred_time = model.predict(X)[0]
    return pred_time  # We want to minimize lap time

def optimize_setup():
    print("🔍 Running Bayesian Optimization...")
    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=40,
        random_state=42
    )

    best_params = dict(zip([dim.name for dim in space], result.x))
    print("✅ Best Setup Found:")
    for k, v in best_params.items():
        print(f"{k}: {v:.2f}")
    print(f"Predicted Lap Time: {result.fun:.2f} sec")

    return best_params, result.fun

if __name__ == "__main__":
    optimize_setup()