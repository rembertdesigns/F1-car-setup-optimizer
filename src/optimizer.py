import joblib
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.ensemble import BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize as nsga_minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
import json

# --- Load Trained ML Model ---
try:
    model = joblib.load("models/lap_time_predictor_v2.pkl")
    with open("models/model_features.json") as f:
        MODEL_FEATURES = json.load(f)
except FileNotFoundError:
    print("Optimizer Error: models/lap_time_predictor_v2.pkl not found.")
    model = None

# --- Define Search Space ---
space = [
    Integer(0, 50, name="front_wing_angle"),
    Integer(0, 50, name="rear_wing_angle"),
    Integer(30, 50, name="ride_height"),
    Integer(1, 11, name="suspension_stiffness"),
    Integer(50, 60, name="brake_bias")
]

param_names = [dim.name for dim in space]

# --- Feature Prep ---
def _prepare_feature_vector(setup_params, track_conditions):
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

# --- Original Bayesian Optimization ---
def optimize_setup(weights, track_conditions):
    if model is None:
        raise RuntimeError("Model not loaded.")

    @use_named_args(space)
    def composite_objective(**setup_params):
        X = _prepare_feature_vector(setup_params, track_conditions)
        predicted_lap_time = model.predict(X)[0]
        tire_penalty = 0.2 * (setup_params["suspension_stiffness"] / 10.0) + 0.1 * abs(setup_params["brake_bias"] - 55) / 5.0
        imbalance = abs(setup_params["rear_wing_angle"] - setup_params["front_wing_angle"]) / 50.0
        score = (weights["lap_time"] * predicted_lap_time +
                 weights["tire_preservation"] * tire_penalty +
                 weights["handling_balance"] * imbalance)
        return score

    result = gp_minimize(func=composite_objective, dimensions=space, n_calls=50, random_state=42, n_jobs=-1)
    best_params = {dim.name: val for dim, val in zip(space, result.x)}
    best_X = _prepare_feature_vector(best_params, track_conditions)
    predicted_lap = model.predict(best_X)[0]
    return best_params, predicted_lap

# --- NSGA-II Multi-Objective ---
class F1SetupNSGA(ElementwiseProblem):
    def __init__(self, track_conditions):
        super().__init__(n_var=5, n_obj=2, n_constr=0,
                         xl=np.array([0, 0, 30, 1, 50]),
                         xu=np.array([50, 50, 50, 11, 60]))
        self.track_conditions = track_conditions

    def _evaluate(self, x, out, *args, **kwargs):
        setup = dict(zip(param_names, x))
        X = _prepare_feature_vector(setup, self.track_conditions)
        lap_time = model.predict(X)[0]
        tire_penalty = 0.2 * (setup["suspension_stiffness"] / 10.0) + 0.1 * abs(setup["brake_bias"] - 55) / 5.0
        out["F"] = [lap_time, tire_penalty]

def run_nsga2(track_conditions):
    problem = F1SetupNSGA(track_conditions)
    algo = NSGA2(pop_size=50)
    termination = get_termination("n_gen", 40)
    res = nsga_minimize(problem, algo, termination, verbose=False)
    results = []
    for i in range(len(res.F)):
        setup = dict(zip(param_names, res.X[i]))
        results.append({"lap_time": res.F[i][0], "tire_penalty": res.F[i][1], "setup": setup})
    return pd.DataFrame(results)

# --- GP Ensemble + Active Learning ---
def load_gp_ensemble(X_train, y_train):
    base_gp = GaussianProcessRegressor()
    ensemble = BaggingRegressor(base_estimator=base_gp, n_estimators=5, random_state=42)
    ensemble.fit(X_train, y_train)
    return ensemble

def predict_with_uncertainty(gp_ensemble, X):
    preds = [est.predict(X) for est in gp_ensemble.estimators_]
    mean = np.mean(preds, axis=0)
    std = np.std(preds, axis=0)
    return mean, std

def active_select_next(gp_ensemble, pool_X, top_k=5):
    preds = np.array([est.predict(pool_X) for est in gp_ensemble.estimators_])
    std = np.std(preds, axis=0)
    top_indices = np.argsort(std)[-top_k:]
    return pool_X.iloc[top_indices]

# --- Pareto Scan Using Original Optimizer ---
def run_pareto_optimization(track_conditions, n_steps=15):
    if model is None:
        raise RuntimeError("Model not loaded.")

    pareto_front = []
    weight_steps = np.linspace(0, 1, n_steps)

    for lap_time_focus in weight_steps:
        weights = {"lap_time": lap_time_focus, "tire_preservation": 1.0 - lap_time_focus, "handling_balance": 0.0}
        best_params, _ = optimize_setup(weights, track_conditions)
        X = _prepare_feature_vector(best_params, track_conditions)
        lap_time = model.predict(X)[0]
        tire_penalty = 0.2 * (best_params["suspension_stiffness"] / 10.0) + 0.1 * abs(best_params["brake_bias"] - 55) / 5.0
        tire_score = 10 - (tire_penalty * 20)
        pareto_front.append({"lap_time": lap_time, "tire_preservation": np.clip(tire_score, 0, 10), "setup": best_params})

    return pd.DataFrame(pareto_front)

# --- TOGGLE FUNCTION ---
def run_optimizer(mode, track_conditions, weights=None):
    if mode == "bayesian":
        return optimize_setup(weights, track_conditions)
    elif mode == "pareto":
        return run_pareto_optimization(track_conditions)
    elif mode == "nsga2":
        return run_nsga2(track_conditions)
    else:
        raise ValueError("Unknown optimization mode. Choose 'bayesian', 'pareto', or 'nsga2'.")
