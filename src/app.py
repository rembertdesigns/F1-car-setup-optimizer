import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
import os
import shap
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from optimizer import run_optimizer, run_pareto_optimization
from sklearn.ensemble import RandomForestRegressor
from physics_model import compute_downforce, compute_drag, compute_brake_distance, simulate_straight, simulate_corner, get_car_params, track
from physics_model import simulate_lap
import joblib
import json
from stable_baselines3 import PPO

try:
    rl_model = PPO.load("models/rl/ppo_car_setup_final.zip")
except Exception as e:
    rl_model = None
    print(f"RL Agent not loaded: {e}")

# Load anomaly detection model
try:
    anomaly_model = joblib.load("models/setup_anomaly_detector.pkl")
    with open("models/anomaly_features.json", "r") as f:
        anomaly_features = json.load(f)
except Exception as e:
    anomaly_model = None
    anomaly_features = []
    print(f"Warning: Anomaly detection model not loaded: {e}")

# Load Predictive Maintenance model
try:
    maintenance_model = joblib.load("models/maintenance_risk_predictor.pkl")
    with open("models/maintenance_features.json", "r") as f:
        maintenance_features = json.load(f)
except Exception as e:
    maintenance_model = None
    maintenance_features = []
    print(f"Warning: Maintenance model not loaded: {e}")

# ✅ Setup Logger Utility
from datetime import datetime

def log_generated_setup(setup, lap_time, source="RL Agent", log_file="data/setup_log.csv"):
    import csv
    os.makedirs("data", exist_ok=True)
    fieldnames = ["timestamp", "source", "lap_time"] + list(setup.keys())
    log_exists = os.path.exists(log_file)

    with open(log_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not log_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.utcnow().isoformat(),
            "source": source,
            "lap_time": lap_time,
            **setup
        })

st.set_page_config(page_title="F1 Car Setup Optimizer", layout="wide")

# --- 🏎️ Title & Intro ---
historic_data = pd.read_csv("data/historic_setups.csv") if os.path.exists("data/historic_setups.csv") else pd.DataFrame()

st.title("🏎️ F1 Car Setup Workbench")
st.markdown("Interactively create and optimize a car setup for different performance tradeoffs.")
# Anchor for return-to-top functionality
st.markdown('<a name="top"></a>', unsafe_allow_html=True)

TRACKS_DATA = {
    "Monza": {
        "description": "The 'Temple of Speed'. Low downforce and top speed dominate.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/ItalianGP/Monza_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 80.0
    },
    "Monaco": {
        "description": "Tight, twisty, and unforgiving. Max downforce and precision needed.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/MonacoGP/Monaco_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 71.0
    },
    "Spa-Francorchamps": {
        "description": "High-speed corners + elevation. A balanced setup is key.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/BelgianGP/Spa-Francorchamps_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 105.0
    },
    "Silverstone": {
        "description": "High-speed flowing track. Requires downforce and balance.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/BritishGP/Silverstone_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 93.0
    },
    "Suzuka": {
        "description": "Figure-8 layout with fast esses. Demands precision and balance.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/JapaneseGP/Suzuka_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 90.0
    },
    "Circuit de Barcelona-Catalunya": {
        "description": "Technical with variety. Good all-round car performance needed.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/SpanishGP/Barcelona_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 91.0
    },
    "Bahrain International Circuit": {
        "description": "Brake-heavy. Good traction and cooling are critical.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/BahrainGP/Bahrain_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 93.5
    },
    "Interlagos": {
        "description": "Short lap. High elevation, quick corners, and overtaking.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/BrazilGP/Interlagos_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 72.5
    },
    "Red Bull Ring": {
        "description": "Short, fast, and flowing. Elevation changes matter.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/AustrianGP/Red_Bull_Ring_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 66.5
    },
    "Hungaroring": {
        "description": "Twisty and technical. High downforce is key.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/HungarianGP/Hungaroring_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 77.5
    },
    "Circuit of the Americas": {
        "description": "Modern layout with elevation. Balanced car is essential.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/USGP/Austin_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 97.0
    },
    "Singapore": {
        "description": "Long, hot night race. Traction and cooling needed.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/SingaporeGP/Singapore_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 110.0
    },
    "Zandvoort": {
        "description": "Fast and narrow. Banked turns and flowing rhythm.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/DutchGP/Zandvoort_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 72.0
    },
    "Jeddah Street Circuit": {
        "description": "Very high speed for a street circuit. Thin margins.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/SaudiArabianGP/Jeddah_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 89.0
    },
    "Las Vegas Strip Circuit": {
        "description": "New layout. Very long straights and low temperatures.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/LasVegasGP/LasVegas_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 100.0
    },
    "Yas Marina (Abu Dhabi)": {
        "description": "Twilight race. Long straights with tight corners.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/AbuDhabiGP/Yas_Marina_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 94.0
    },
    "Imola": {
        "description": "Historic, fast, and narrow. Track limits matter.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/EmiliaRomagnaGP/Imola_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 80.0
    },
    "Baku City Circuit": {
        "description": "Huge straights + tight corners. Risk and reward.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/AzerbaijanGP/Baku_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 102.0
    },
    "Shanghai": {
        "description": "Modern layout. Long corners test car stability.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/ChineseGP/Shanghai_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 95.0
    },
    "Canada (Montreal)": {
        "description": "Challenging braking zones and walls. Low drag helps.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/CanadianGP/Montreal_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 76.0
    },
    "Miami": {
        "description": "New track. High-speed corners and elevation changes.",
        "image": "https://www.formula1.com/content/dam/fom-website/manual/Misc/2023manual/MiamiGP/Miami_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 90.0
    }
}

col1, col2, col3 = st.columns([1, 1, 1], gap="large")

# --- ⚙️ Controls ---
with col1:
    st.header("⚙️ Controls")

if 'setup' not in st.session_state:
    st.session_state.setup = {"front_wing_angle": 25, "rear_wing_angle": 25, "ride_height": 35, "suspension_stiffness": 5, "brake_bias": 54}
if 'setup_A' not in st.session_state:
    st.session_state.setup_A = None
if 'setup_B' not in st.session_state:
    st.session_state.setup_B = None
if 'pareto_front' not in st.session_state:
    st.session_state.pareto_front = None

selected_track = st.selectbox("Select Track", list(TRACKS_DATA.keys()))
track_info = TRACKS_DATA[selected_track]

st.image(track_info["image"], use_container_width=True)
st.info(track_info["description"])

track_conditions = {
    "base_lap_time": track_info.get("base_lap_time", 90.0),
    "track_temperature": st.slider("Track Temperature (°C)", 15.0, 50.0, track_info.get("track_temperature", 30.0)),
    "grip_level": st.slider("Grip Level", 0.8, 1.2, track_info.get("grip_level", 1.0), 0.01),
    "tire_compound": st.selectbox("Target Tire Compound", ["soft", "medium", "hard"], index=0)
}

# --- 🌦️ Weather & Fuel Inputs ---
weather = st.selectbox("Weather Conditions", ["Dry", "Light Rain", "Heavy Rain"], index=0)
initial_fuel = st.slider("Initial Fuel Load (kg)", 80, 110, 100, step=1)

# --- 🔀 Optimizer Mode Selector ---
st.divider()
st.subheader("🔁 Choose Optimization Mode")
mode = st.selectbox(
    "Select an Optimization Mode",
    ["Bayesian Optimization", "Pareto Scan", "NSGA-II Multi-Objective"],
    index=0
)

mode_map = {
    "Bayesian Optimization": "bayesian",
    "Pareto Scan": "pareto",
    "NSGA-II Multi-Objective": "nsga2"
}

selected_mode = mode_map[mode]

# --- ⚖️ Optimization Weights ---
with col3:
    st.header("⚖️ Optimization Weights")
    lap_time_weight = st.slider("🏁 Lap Time Focus", 0.0, 1.0, 0.6, 0.05, key="weight_lap")
    tire_preservation_weight = st.slider("🛞 Tire Preservation", 0.0, 1.0, 0.2, 0.05, key="weight_tire")
    handling_weight = st.slider("↔️ Handling Balance", 0.0, 1.0, 0.2, 0.05, key="weight_handling")
    total_weight = lap_time_weight + tire_preservation_weight + handling_weight
    if not np.isclose(total_weight, 1.0):
        st.error(f"Weights must sum to 1.0 (is {total_weight:.2f})")

# --- COLUMN 2: SETUP WORKBENCH ---
with col2:
    st.header("🔧 Setup Workbench")
    st.markdown("Adjust sliders to create a setup, or run an optimizer to find one.")
    
    setup = st.session_state.setup
    setup["front_wing_angle"] = st.slider("Front Wing Angle", 0, 50, setup["front_wing_angle"])
    setup["rear_wing_angle"] = st.slider("Rear Wing Angle", 0, 50, setup["rear_wing_angle"])
    setup["ride_height"] = st.slider("Ride Height (mm)", 30, 50, setup["ride_height"])
    setup["suspension_stiffness"] = st.slider("Suspension Stiffness", 1, 11, setup["suspension_stiffness"])
    setup["brake_bias"] = st.slider("Brake Bias (%)", 50, 60, setup["brake_bias"])
    
    st.divider()
    st.subheader("🎯 Actions")
    c1, c2, c3 = st.columns(3)

if c1.button("Run Optimization", use_container_width=True, type="primary", disabled=not np.isclose(total_weight, 1.0)):
    weights = {
        "lap_time": lap_time_weight,
        "tire_preservation": tire_preservation_weight,
        "handling_balance": handling_weight
    }

    with st.spinner(f"Running {mode}..."):
        result = run_optimizer(selected_mode, track_conditions, weights=weights if selected_mode == "bayesian" else None)

    if selected_mode == "bayesian":
        best_params, predicted_lap = result
        st.success(f"Optimized Lap Time: {predicted_lap:.3f}s")
        for k, v in best_params.items():
            st.session_state.setup[k] = int(v)
    else:
        st.session_state.pareto_front = result
        st.success(f"{len(result)} results found for {mode}!")
        st.dataframe(result)

    st.rerun()


if c2.button("Save to Slot A", use_container_width=True):
    st.session_state.setup_A = st.session_state.setup.copy()
    st.toast("✅ Setup A saved!")

if c3.button("Save to Slot B", use_container_width=True):
    if st.session_state.setup != st.session_state.get("setup_A", {}):
        st.session_state.setup_B = st.session_state.setup.copy()
        st.toast("✅ Setup B saved!")
    else:
        st.warning("Setup B is identical to Setup A. Try adjusting it first.")

# 🆕 Show delta from Setup A before saving B
if st.session_state.get("setup_A"):
    delta = {
        k: st.session_state.setup[k] - st.session_state.setup_A[k]
        for k in st.session_state.setup
    }
    st.caption("Δ from Setup A:")
    st.json(delta)

def get_balance_scores(setup_dict):
    top_speed = 10 - ((setup_dict["front_wing_angle"] + setup_dict["rear_wing_angle"]) / 100) * 10
    cornering = ((setup_dict["front_wing_angle"] + setup_dict["rear_wing_angle"]) / 100) * 8 + (setup_dict["suspension_stiffness"] / 11) * 2
    stability = ((50 - setup_dict["ride_height"]) / 20) * 5 + (setup_dict["brake_bias"] / 60) * 5
    tire_life = 10 - ((setup_dict["suspension_stiffness"] / 11) * 5 + top_speed * 0.1)
    return [np.clip(s, 0, 10) for s in [top_speed, cornering, stability, tire_life]]

# Run RL Agent Button
if rl_model and st.button("🤖 Generate Setup with RL Agent", use_container_width=True):
    obs = np.array([
        st.session_state.setup["front_wing_angle"],
        st.session_state.setup["rear_wing_angle"],
        st.session_state.setup["ride_height"],
        st.session_state.setup["suspension_stiffness"],
        st.session_state.setup["brake_bias"]
    ], dtype=np.float32)

    action, _ = rl_model.predict(obs, deterministic=True)

    # Apply action like in env
    low = np.array([0, 0, 30, 1, 50])
    high = np.array([50, 50, 50, 11, 60])
    delta = (high - low) * 0.05 * action
    new_setup = np.clip(obs + delta, low, high)

    keys = ["front_wing_angle", "rear_wing_angle", "ride_height", "suspension_stiffness", "brake_bias"]
    for i, k in enumerate(keys):
        st.session_state.setup[k] = int(round(new_setup[i]))

    st.success("✅ Setup generated by RL agent!")
    st.rerun()

# --- Anomaly Detection Helper ---
def check_for_anomaly(setup_dict, track_conditions, fuel_weight=100, traffic=0.5):
    if anomaly_model is None:
        return False, None
    try:
        full_input = {
            **setup_dict,
            "track_temperature": track_conditions.get("track_temperature", 30.0),
            "grip_level": track_conditions.get("grip_level", 1.0),
            "fuel_weight": fuel_weight,
            "traffic": traffic
        }
        X = pd.DataFrame([full_input])[anomaly_features]
        is_anomaly = anomaly_model.predict(X)[0] == -1
        return is_anomaly, X.to_dict(orient="records")[0]
    except Exception as e:
        print(f"Anomaly check failed: {e}")
        return False, None
    
# --- Predictive Maintenance Helper ---
def predict_maintenance_risk(setup_dict, track_conditions):
    if maintenance_model is None:
        return None
    try:
        input_data = {
            "front_wing_angle": setup_dict["front_wing_angle"],
            "rear_wing_angle": setup_dict["rear_wing_angle"],
            "ride_height": setup_dict["ride_height"],
            "suspension_stiffness": setup_dict["suspension_stiffness"],
            "brake_bias": setup_dict["brake_bias"],
            "track_temperature": track_conditions["track_temperature"],
            "grip_level": track_conditions["grip_level"]
        }
        df_input = pd.DataFrame([input_data])[maintenance_features]

        # ✅ Access the first (and only) prediction
        prediction = maintenance_model.predict(df_input)
        return float(prediction[0])  # Ensure it's a float for Streamlit metric
    except Exception as e:
        print(f"Maintenance prediction failed: {e}")
        return None

# --- 📊 Live Analysis ---
st.divider()
st.header("📊 Live Analysis")

# --- Anomaly Detection ---
is_anomaly, input_data = check_for_anomaly(
    st.session_state.setup,
    track_conditions=track_conditions,
    fuel_weight=initial_fuel,
    traffic=0.5  # You can later expose this as a slider
)
if is_anomaly:
    st.error("⚠️ This setup appears to be an **outlier** or **potentially unsafe configuration**. Please review parameters carefully.")

scores = get_balance_scores(st.session_state.setup)
categories = ['Top Speed', 'Cornering Grip', 'Stability', 'Tire Life']
fig = go.Figure(data=go.Scatterpolar(r=scores, theta=categories, fill='toself', name='Current Setup', line_color='#00D2BE'))
fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=False, height=300)
st.plotly_chart(fig, use_container_width=True)

def generate_telemetry_trace(setup_dict):
    total_distance = 5000
    distance = np.linspace(0, total_distance, 500)
    straight1, corner1, straight2, corner2, straight3 = 1000, 1500, 3000, 3500, total_distance
    max_s = 350 - (setup_dict["front_wing_angle"] + setup_dict["rear_wing_angle"]) * 0.5
    min_s = 80 + (setup_dict["front_wing_angle"] + setup_dict["rear_wing_angle"]) * 0.3
    fast_s = 200 + (setup_dict["front_wing_angle"] + setup_dict["rear_wing_angle"]) * 0.4
    speed = np.zeros_like(distance)
    speed[distance <= straight1] = max_s
    speed[(distance > straight1) & (distance <= corner1)] = min_s
    speed[(distance > corner1) & (distance <= straight2)] = max_s
    speed[(distance > straight2) & (distance <= corner2)] = fast_s
    speed[distance > corner2] = max_s
    speed = np.convolve(speed, np.ones(15)/15, mode='same')
    return distance, speed

distance, speed = generate_telemetry_trace(st.session_state.setup)
fig_telemetry = go.Figure(data=go.Scatter(x=distance, y=speed, mode='lines', name='Speed', line=dict(color='#FF8700', width=4)))
fig_telemetry.update_layout(title="Simulated Speed Trace", xaxis_title="Distance (m)", yaxis_title="Speed (km/h)", height=250)
st.plotly_chart(fig_telemetry, use_container_width=True)

# --- 🔧 Predictive Maintenance Risk ---
maintenance_risk = predict_maintenance_risk(st.session_state.setup, track_conditions)
if maintenance_risk is not None:
    st.subheader("🔧 Predicted Maintenance Risk")
    st.metric("Component Wear Risk", f"{maintenance_risk:.1f}%")
    if maintenance_risk > 60:
        st.warning("⚠️ High risk of component wear detected. Consider adjusting setup.")

from physics_model import simulate_lap

# Simulate lap with enhanced physics
lap_time, forces = simulate_lap(
    st.session_state.setup,
    ambient_temp=track_conditions["track_temperature"],
    fuel_start=initial_fuel,
    weather=weather
)

with st.expander("🌡️ Tire Temperature Over Lap", expanded=False):
    temps = [f["tire_temp"] for f in forces if "tire_temp" in f]
    segments = list(range(1, len(temps) + 1))
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=segments, y=temps, mode="lines+markers", name="Tire Temp (°C)"))
    fig_temp.update_layout(title="Tire Temperature Evolution", xaxis_title="Segment", yaxis_title="Temp (°C)")
    st.plotly_chart(fig_temp, use_container_width=True)


# --- NEW: Physics Explorer Section ---
st.divider()
st.header("📐 Physics Explorer")
st.caption("Visualize core physics forces like aero downforce, drag, and brake distance based on current setup.")

if 'setup' not in st.session_state:
    st.session_state.setup = {
        "front_wing_angle": 25,
        "rear_wing_angle": 25,
        "ride_height": 35,
        "suspension_stiffness": 5,
        "brake_bias": 54
    }

setup = st.session_state.setup

# Input ranges
speeds = np.linspace(50, 350, 100)  # km/h

# Downforce
car = get_car_params(setup)
downforces = [compute_downforce(speed, car["downforce_coeff"]) for speed in speeds]
fig_df = go.Figure()
fig_df.add_trace(go.Scatter(x=speeds, y=downforces, mode='lines', name='Downforce'))
fig_df.update_layout(title="Aero Downforce vs. Speed", xaxis_title="Speed (km/h)", yaxis_title="Downforce (N)")
st.plotly_chart(fig_df, use_container_width=True)

# Drag Force
car = get_car_params(setup)
drags = [compute_drag(speed, car["drag_coeff"]) for speed in speeds]
fig_drag = go.Figure()
fig_drag.add_trace(go.Scatter(x=speeds, y=drags, mode='lines', name='Drag Force', line=dict(color='orange')))
fig_drag.update_layout(title="Drag Force vs. Speed", xaxis_title="Speed (km/h)", yaxis_title="Drag (N)")
st.plotly_chart(fig_drag, use_container_width=True)

# Brake Distance (at different entry speeds)
st.subheader("🛑 Brake Distance at Various Speeds")
brake_speeds = [100, 150, 200, 250, 300]
brake_distances = [compute_brake_distance(v) for v in brake_speeds]
st.bar_chart(pd.DataFrame({"Speed (km/h)": brake_speeds, "Brake Distance (m)": brake_distances}).set_index("Speed (km/h)"))

def get_predicted_lap_time(setup_dict, track_conditions):
    """Calculates a predicted lap time based on a simple formula for display."""
    base_lap = track_conditions.get('base_lap_time', 90.0)
    aero_penalty = (setup_dict["front_wing_angle"] + setup_dict["rear_wing_angle"]) * 0.04
    cornering_bonus = (setup_dict["front_wing_angle"] + setup_dict["rear_wing_angle"]) * 0.02
    ride_height_effect = (setup_dict["ride_height"] - 30) * 0.05
    return base_lap + aero_penalty - cornering_bonus + ride_height_effect

def check_for_anomaly(setup_dict, track_conditions):
    if not anomaly_model:
        return False
    feature_input = {
        "front_wing_angle": setup_dict["front_wing_angle"],
        "rear_wing_angle": setup_dict["rear_wing_angle"],
        "ride_height": setup_dict["ride_height"],
        "suspension_stiffness": setup_dict["suspension_stiffness"],
        "brake_bias": setup_dict["brake_bias"],
        "track_temperature": track_conditions.get("track_temperature", 30.0),
        "grip_level": track_conditions.get("grip_level", 1.0),
        "fuel_weight": track_conditions.get("fuel_weight", 10.0),
        "traffic": track_conditions.get("traffic", 0.2)
    }
    X = pd.DataFrame([feature_input])[anomaly_features]
    return anomaly_model.predict(X)[0] == -1  # -1 = anomaly

# --- 📊 Sensitivity Analysis ---
with st.expander("📊 Sensitivity Analysis", expanded=False):
    st.header("📊 Sensitivity Analysis")

    st.markdown("Understand how changes in individual setup parameters affect lap time.")

    selected_param = st.selectbox("Choose a parameter to analyze:", [
        "front_wing_angle",
        "rear_wing_angle",
        "ride_height",
        "suspension_stiffness",
        "brake_bias"
    ], key="sensitivity_param")

    param_range = {
        "front_wing_angle": range(0, 51, 5),
        "rear_wing_angle": range(0, 51, 5),
        "ride_height": range(20, 51, 2),
        "suspension_stiffness": range(1, 11),
        "brake_bias": range(50, 61)
    }

    setup_copy = st.session_state.setup.copy()
    x_vals, y_vals = [], []
    for val in param_range[selected_param]:
        setup_copy[selected_param] = val
        lap = get_predicted_lap_time(setup_copy, track_conditions)
        x_vals.append(val)
        y_vals.append(lap)

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines+markers", line=dict(width=3)))
    fig_sens.update_layout(
        title=f"Lap Time Sensitivity to {selected_param.replace('_',' ').title()}",
        xaxis_title=selected_param.replace('_',' ').title(),
        yaxis_title="Predicted Lap Time (s)",
        height=400
    )
    st.plotly_chart(fig_sens, use_container_width=True)

# --- 🎥 Telemetry Playback Viewer ---
with st.expander("🎥 Telemetry Playback Viewer", expanded=False):
    st.header("🎥 Telemetry Playback Viewer")

    st.markdown("Visualize how your current setup performs across different track segments.")

    def generate_telemetry_trace(setup):
        distance = np.linspace(0, 5000, 500)
        max_speed = 350 - (setup["front_wing_angle"] + setup["rear_wing_angle"]) * 0.5
        min_speed = 80 + (setup["front_wing_angle"] + setup["rear_wing_angle"]) * 0.3
        fast_speed = 200 + (setup["front_wing_angle"] + setup["rear_wing_angle"]) * 0.4

        speed = np.zeros_like(distance)
        speed[distance <= 1000] = max_speed
        speed[(distance > 1000) & (distance <= 1500)] = min_speed
        speed[(distance > 1500) & (distance <= 3000)] = max_speed
        speed[(distance > 3000) & (distance <= 3500)] = fast_speed
        speed[distance > 3500] = max_speed

        speed = np.convolve(speed, np.ones(15)/15, mode='same')
        return distance, speed

    dist, spd = generate_telemetry_trace(st.session_state.setup)
    fig_playback = go.Figure()
    fig_playback.add_trace(go.Scatter(x=dist, y=spd, mode='lines', name='Speed Trace'))
    fig_playback.update_layout(
        title="Simulated Speed Trace Based on Setup",
        xaxis_title="Distance (m)",
        yaxis_title="Speed (km/h)",
        height=400
    )
    st.plotly_chart(fig_playback, use_container_width=True)

# --- Interactive Track Walkthrough + Setup Notes ---
st.header("🔹 Track Walkthrough & Sector Notes")
st.markdown("Click a sector and leave a setup-related comment.")

track_sectors = ["Sector 1", "Sector 2", "Sector 3"]
selected_sector = st.selectbox("Choose Sector", track_sectors)
comment_key = f"comment_{selected_sector.replace(' ', '_')}"
if comment_key not in st.session_state:
    st.session_state[comment_key] = ""
st.session_state[comment_key] = st.text_area(f"Comment for {selected_sector}", value=st.session_state[comment_key], height=100)

# Display all comments
with st.expander("📃 View All Sector Notes"):
    for sector in track_sectors:
        key = f"comment_{sector.replace(' ', '_')}"
        if st.session_state.get(key):
            st.markdown(f"**{sector}**: {st.session_state[key]}")

# --- 📊 Segment-Level Physics Breakdown ---
st.divider()
st.header("📊 Segment-Level Physics Breakdown")

segment_data = []
v = 0  # Initial speed

car = get_car_params(st.session_state.setup)

for i, segment in enumerate(track):
    if segment["type"] == "straight":
        length = segment["length"]
        next_v, t = simulate_straight(length, v, car)
        segment_data.append({
            "Segment": f"S{i+1} - Straight",
            "Length (m)": length,
            "Entry Speed (km/h)": round(v, 1),
            "Exit Speed (km/h)": round(next_v, 1),
            "Time (s)": round(t, 2)
        })
        v = next_v
    elif segment["type"] == "corner":
        radius = segment["radius"]
        angle = segment["angle"]
        next_v, t = simulate_corner(radius, angle, v, car)
        segment_data.append({
            "Segment": f"S{i+1} - Corner",
            "Radius (m)": radius,
            "Angle (deg)": angle,
            "Entry Speed (km/h)": round(v, 1),
            "Corner Speed (km/h)": round(next_v, 1),
            "Time (s)": round(t, 2)
        })
        v = next_v

st.dataframe(pd.DataFrame(segment_data))

# --- 🧮 Pareto Optimization (Performance Tradeoff Analysis) ---
with st.expander("🧮 Setup Optimization & Tradeoff Analysis", expanded=False):
    st.header("🧮 Setup Optimization & Tradeoff Analysis")

    if st.button("Run Optimization", key="btn_run_optimization"):
        st.toast("Running multi-objective optimization...", icon="🚀")
        pareto_front = run_pareto_optimization(track_conditions)
        st.session_state.pareto_front = pareto_front
        st.success(f"Found {len(pareto_front)} Pareto-optimal setups.")

    if st.session_state.pareto_front:
        st.subheader("📈 Pareto Front")
        df = pd.DataFrame(st.session_state.pareto_front)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['lap_time'],
            y=df['tire_wear'],
            mode='markers+text',
            marker=dict(size=10, color='orange'),
            text=[f"#{i+1}" for i in range(len(df))],
            textposition="top center"
        ))
        fig.update_layout(
            xaxis_title="Lap Time (s)",
            yaxis_title="Tire Wear Index",
            title="Lap Time vs Tire Wear Tradeoff",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        sel = st.selectbox("Select a setup to load into workbench:", options=list(range(len(df))), format_func=lambda i: f"Setup #{i+1}" if i is not None else "None")
        if st.button("Load Selected Setup", key="btn_load_pareto_setup"):
            chosen = df.iloc[sel][['front_wing_angle', 'rear_wing_angle', 'ride_height', 'suspension_stiffness', 'brake_bias']].astype(int).to_dict()
            for k, v in chosen.items():
                st.session_state.setup[k] = v
            st.success(f"Loaded Setup #{sel+1} into Workbench!")
            st.rerun()

# --- ⚖️ Setup Comparison Workbench ---
st.divider()
st.header("⚖️ Setup Comparison Workbench")

if st.session_state.setup_A is None and st.session_state.setup_B is None:
    st.info("Use the 'Save to Slot' buttons in the workbench above to compare setups.")
else:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.subheader("Setup A")
        if st.session_state.setup_A:
            lap_time_A = get_predicted_lap_time(st.session_state.setup_A, track_conditions)
            st.metric("Predicted Lap Time", f"{lap_time_A:.3f}s")
        else:
            st.info("No setup in Slot A.")

    with c2:
        st.subheader("Setup B")
        if st.session_state.setup_B:
            lap_time_B = get_predicted_lap_time(st.session_state.setup_B, track_conditions)
            delta = lap_time_B - get_predicted_lap_time(st.session_state.setup_A, track_conditions) if st.session_state.setup_A else None
            st.metric("Predicted Lap Time", f"{lap_time_B:.3f}s", delta=f"{delta:.3f}s" if delta is not None else None, delta_color="inverse")
        else:
            st.info("No setup in Slot B.")

    # Flattened Expanders (not inside another expander)
    if st.session_state.setup_A:
        with st.expander("📋 Show Setup A Parameters"):
            st.json(st.session_state.setup_A)

    if st.session_state.setup_B:
        with st.expander("📋 Show Setup B Parameters"):
            st.json(st.session_state.setup_B)

    if st.session_state.setup_A and st.session_state.setup_B:
        st.subheader("🛰️ Comparative Telemetry & Balance Overlay")

        fig_comp_tele = go.Figure()
        dist_A, speed_A = generate_telemetry_trace(st.session_state.setup_A)
        fig_comp_tele.add_trace(go.Scatter(x=dist_A, y=speed_A, mode='lines', name='Setup A', line=dict(color='cyan', width=4)))
        dist_B, speed_B = generate_telemetry_trace(st.session_state.setup_B)
        fig_comp_tele.add_trace(go.Scatter(x=dist_B, y=speed_B, mode='lines', name='Setup B', line=dict(color='magenta', width=4, dash='dot')))
        fig_comp_tele.update_layout(title="Speed Trace Comparison", xaxis_title="Distance (m)", yaxis_title="Speed (km/h)", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

        fig_comp_radar = go.Figure()
        categories = ['Top Speed', 'Cornering Grip', 'Stability', 'Tire Life']
        scores_A = get_balance_scores(st.session_state.setup_A)
        fig_comp_radar.add_trace(go.Scatterpolar(r=scores_A, theta=categories, fill='toself', name='Setup A', line_color='cyan', opacity=0.7))
        scores_B = get_balance_scores(st.session_state.setup_B)
        fig_comp_radar.add_trace(go.Scatterpolar(r=scores_B, theta=categories, fill='toself', name='Setup B', line_color='magenta', opacity=0.7))
        fig_comp_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), title="Setup Balance Comparison", legend=dict(yanchor="bottom", y=0, xanchor="left", x=0))

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_comp_tele, use_container_width=True)
        with c2:
            st.plotly_chart(fig_comp_radar, use_container_width=True)

# --- 🤖 Recommended Setup ---
st.divider()
st.header("🤖 Recommended Setup Based on Conditions")

# Train the recommender model only once
@st.cache_data
def train_recommender(data):
    if data.empty:
        return None, None
    features = ["front_wing_angle", "rear_wing_angle", "ride_height", "suspension_stiffness", "brake_bias"]
    X = data[features]
    y = data["lap_time"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    explainer = shap.Explainer(model, X)
    return model, explainer

recommender_model, shap_explainer = train_recommender(historic_data)

# Generate recommendation based on closest historic match
def recommend_setup(track_conditions):
    if recommender_model is None:
        return None
    track_temp = track_conditions["track_temperature"]
    grip = track_conditions["grip_level"]
    target_row = historic_data.copy()
    target_row["score"] = np.abs(historic_data["track_temperature"] - track_temp) + np.abs(historic_data["grip_level"] - grip)
    closest = target_row.sort_values("score").iloc[0]
    return {
        "front_wing_angle": int(closest["front_wing_angle"]),
        "rear_wing_angle": int(closest["rear_wing_angle"]),
        "ride_height": int(closest["ride_height"]),
        "suspension_stiffness": int(closest["suspension_stiffness"]),
        "brake_bias": int(closest["brake_bias"])
    }

# --- 🤖 Recommended Setup ---
with st.expander("🤖 Recommended Setup Based on Conditions", expanded=False):
    st.header("🤖 Recommended Setup Based on Conditions")

    if st.button("Get Recommended Setup", key="btn_recommended_setup"):
        rec = recommend_setup(track_conditions)
        if rec:
            st.success("Recommended setup loaded into workbench!")
            for k, v in rec.items():
                st.session_state.setup[k] = v
            st.rerun()
        else:
            st.warning("No historic data to make recommendation.")

# --- 📤 Export & Share ---
with st.expander("📤 Export & Share", expanded=False):
    st.header("📤 Export & Share")
    st.markdown("Download the current setup from the workbench or generate a shareable link.")

    current_setup = st.session_state.setup
    base_lap = track_conditions.get("base_lap_time", 90.0)
    aero_penalty = (current_setup["front_wing_angle"] + current_setup["rear_wing_angle"]) * 0.04
    cornering_bonus = (current_setup["front_wing_angle"] + current_setup["rear_wing_angle"]) * 0.02
    ride_height_effect = (current_setup["ride_height"] - 30) * 0.05
    current_lap_time = base_lap + aero_penalty - cornering_bonus + ride_height_effect

    # PDF Generator
    def generate_setup_pdf(setup_dict, track_name, lap_time):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "F1 Car Setup Report", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, f"Track: {track_name}", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.ln(10)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Predicted Lap Time:", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 20)
        pdf.cell(0, 10, f"{lap_time:.3f}s", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Setup Parameters:", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 11)
        for key, val in setup_dict.items():
            pdf.cell(0, 7, f"  -  {key.replace('_', ' ').title()}: {val}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        return bytes(pdf.output())

    export_cols = st.columns(3)
    with export_cols[0]:
        json_string = json.dumps(current_setup, indent=4)
        st.download_button(
            label="Download as JSON",
            file_name=f"f1_setup_{selected_track}.json",
            mime="application/json",
            data=json_string,
            use_container_width=True,
            key="btn_download_json"
        )

    with export_cols[1]:
        pdf_bytes = generate_setup_pdf(current_setup, selected_track, current_lap_time)
        st.download_button(
            label="Download as PDF",
            file_name=f"f1_setup_{selected_track}.pdf",
            mime="application/pdf",
            data=pdf_bytes,
            use_container_width=True,
            key="btn_download_pdf"
        )

    base_url = "https://your-app-url.streamlit.app/"  # Replace with your deployed app URL
    query_params = f"?fwa={current_setup['front_wing_angle']}&rwa={current_setup['rear_wing_angle']}&rh={current_setup['ride_height']}&ss={current_setup['suspension_stiffness']}&bb={current_setup['brake_bias']}"
    share_url = base_url + query_params

    with export_cols[2]:
        if st.button("Generate Shareable Link", use_container_width=True, key="btn_generate_link"):
            st.code(share_url, language=None)

# --- Train Recommender ---
def train_recommender(data):
    if data.empty:
        return None, None
    features = ["front_wing_angle", "rear_wing_angle", "ride_height", "suspension_stiffness", "brake_bias"]
    X = data[features]
    y = data["lap_time"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    explainer = shap.Explainer(model, X)
    return model, explainer

recommender_model, shap_explainer = train_recommender(historic_data)

# --- Helper: Recommend Setup ---
def recommend_setup(track_conditions):
    if recommender_model is None:
        return None
    track_temp = track_conditions["track_temperature"]
    grip = track_conditions["grip_level"]
    target_row = historic_data.copy()
    target_row["score"] = np.abs(historic_data["track_temperature"] - track_temp) + np.abs(historic_data["grip_level"] - grip)
    closest = target_row.sort_values("score").iloc[0]
    return {
        "front_wing_angle": int(closest["front_wing_angle"]),
        "rear_wing_angle": int(closest["rear_wing_angle"]),
        "ride_height": int(closest["ride_height"]),
        "suspension_stiffness": int(closest["suspension_stiffness"]),
        "brake_bias": int(closest["brake_bias"])
    }

# --- 🧠 SHAP Explainability ---
with st.expander("🧠 SHAP Explainability", expanded=False):
    st.header("🧠 SHAP Explainability")

    # Reuse the trained SHAP explainer from earlier if available
    def explain_current_setup(current_setup):
        if shap_explainer is None:
            st.warning("Explainability model not available.")
            return
        sample = pd.DataFrame([current_setup])
        shap_values = shap_explainer(sample)
        st.subheader("🔍 Why this setup works")
        st.caption("Feature importance visualized using SHAP values.")
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0.1)

    if st.button("Show SHAP Insights", key="btn_shap_insights"):
        explain_current_setup(st.session_state.setup)

# --- 📘 Setup Log Viewer ---
st.divider()
st.subheader("📘 RL & Manual Setup Log")

log_path = "data/setup_log.csv"
if os.path.exists(log_path):
    df_log = pd.read_csv(log_path)
    st.dataframe(df_log.sort_values("timestamp", ascending=False).reset_index(drop=True))

    # --- 📥 CSV Download ---
    csv = df_log.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Setup Log as CSV",
        data=csv,
        file_name="setup_log.csv",
        mime="text/csv"
    )

# --- 📊 RL vs Manual Lap Time Trends (Colored by Tire Type) ---
if all(col in df_log.columns for col in ["timestamp", "lap_time", "tire_type", "source"]):
    fig = px.line(
        df_log.sort_values("timestamp"),
        x="timestamp",
        y="lap_time",
        color="tire_type",  # Color by compound type: soft, medium, hard, etc.
        line_dash="source",  # Dashed line for RL vs solid for Manual
        markers=True,
        title="Lap Time Trends by Tire Compound"
    )
    fig.update_layout(height=320, legend_title_text="Tire Type")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No logged setups yet. Use the RL agent or optimizer to generate and log a setup.")

# --- Return to Top Button ---
st.markdown("""
    <div style="text-align:right; margin-top: 3em;">
        <a href="#top">
            <button style="padding:0.5em 1em; font-size:1em; border:none; background-color:#2e86de; color:white; border-radius:6px; cursor:pointer;">
                🔝 Return to Top
            </button>
        </a>
    </div>
""", unsafe_allow_html=True)