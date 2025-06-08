import streamlit as st
import plotly.graph_objects as go
import time
import os
import pandas as pd
import numpy as np 

# This now assumes your 'optimizer.py' file is updated and correct.
from optimizer import optimize_setup

# --- Page Setup ---
st.set_page_config(page_title="F1 Car Setup Optimizer", layout="wide")

st.title("🏎️ F1 Car Setup Workbench")
st.markdown("Interactively create and optimize a car setup for different performance tradeoffs.")
st.divider()

# --- Data & Configuration ---
TRACKS_DATA = {
    "Monza": {
        "description": "The 'Temple of Speed'. Requires low downforce for its long straights, but needs stability for chicanes.",
        "image": "https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Monza_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 80.0
    },
    "Monaco": {
        "description": "A tight, twisty street circuit where maximum downforce, agility, and stability are crucial. Top speed is irrelevant.",
        "image": "https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Monoco_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 71.0
    },
    "Spa-Francorchamps": {
        "description": "A classic track with a mix of long straights and fast, flowing corners. Requires a balanced setup.",
        "image": "https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Spa-Francorchamps_Circuit.png.transform/9col-retina/image.png",
        "base_lap_time": 105.0
    }
}

# --- Helper Functions ---
def get_balance_scores(setup_dict):
    """Calculates performance scores based on setup parameters for the radar chart."""
    top_speed = 10 - ((setup_dict["front_wing_angle"] + setup_dict["rear_wing_angle"]) / 100) * 10
    cornering = ((setup_dict["front_wing_angle"] + setup_dict["rear_wing_angle"]) / 100) * 8 + (setup_dict["suspension_stiffness"] / 11) * 2
    stability = ((50 - setup_dict["ride_height"]) / 20) * 5 + (setup_dict["brake_bias"] / 60) * 5
    tire_life = 10 - ((setup_dict["suspension_stiffness"] / 11) * 5 + top_speed * 0.1)
    return [np.clip(s, 0, 10) for s in [top_speed, cornering, stability, tire_life]]

def get_predicted_lap_time(setup_dict, track_conditions):
    """Calculates a predicted lap time based on a simple formula for display."""
    # This mock function mirrors the logic in the enhanced optimizer.py
    base_lap = track_conditions.get('base_lap_time', 90.0)
    aero_penalty = (setup_dict["front_wing_angle"] + setup_dict["rear_wing_angle"]) * 0.04
    cornering_bonus = (setup_dict["front_wing_angle"] + setup_dict["rear_wing_angle"]) * 0.02
    ride_height_effect = (setup_dict["ride_height"] - 30) * 0.05
    return base_lap + aero_penalty - cornering_bonus + ride_height_effect


# --- Initialize Session State ---
if 'setup' not in st.session_state:
    st.session_state.setup = {"front_wing_angle": 25, "rear_wing_angle": 25, "ride_height": 35, "suspension_stiffness": 5, "brake_bias": 54}
if 'setup_A' not in st.session_state:
    st.session_state.setup_A = None
if 'setup_B' not in st.session_state:
    st.session_state.setup_B = None


# --- Sidebar: Optimization Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    st.subheader("📍 Select Track")
    selected_track = st.selectbox("Choose a track for setup optimization", list(TRACKS_DATA.keys()))
    track_info = TRACKS_DATA[selected_track]
    st.image(track_info["image"], use_column_width=True)
    st.info(track_info["description"])
    track_conditions = {"base_lap_time": track_info.get("base_lap_time", 90.0)}
    st.divider()
    st.header("⚖️ Optimization Weights")
    st.markdown("Define what matters most for your optimal setup. Weights must sum to 1.0.")
    lap_time_weight = st.slider("🏁 Lap Time Focus", 0.0, 1.0, 0.6, 0.05)
    tire_preservation_weight = st.slider("🛞 Tire Preservation Focus", 0.0, 1.0, 0.2, 0.05)
    handling_weight = st.slider("↔️ Handling Balance Focus", 0.0, 1.0, 0.2, 0.05)
    total_weight = lap_time_weight + tire_preservation_weight + handling_weight
    if not np.isclose(total_weight, 1.0):
        st.sidebar.error(f"Weights must sum to 1.0 (Current sum: {total_weight:.2f})")

# --- Main UI Layout ---
col1, col2 = st.columns([0.6, 0.4], gap="large")

with col1:
    st.subheader("🔧 Car Setup Parameters")
    st.markdown("Adjust the sliders below to create a custom setup, or use the optimizer to find the best one.")
    
    setup = st.session_state.setup
    setup["front_wing_angle"] = st.slider("Front Wing Angle", 0, 50, setup["front_wing_angle"])
    setup["rear_wing_angle"] = st.slider("Rear Wing Angle", 0, 50, setup["rear_wing_angle"])
    setup["ride_height"] = st.slider("Ride Height (mm)", 30, 50, setup["ride_height"])
    setup["suspension_stiffness"] = st.slider("Suspension Stiffness", 1, 11, setup["suspension_stiffness"])
    setup["brake_bias"] = st.slider("Brake Bias (%)", 50, 60, setup["brake_bias"])
    
    st.divider()

    st.subheader("🎯 Run Optimization & Save Setups")
    opt_col, save_a_col, save_b_col = st.columns(3)

    with opt_col:
        if st.button("Find Optimal Setup", use_container_width=True, type="primary", disabled=not np.isclose(total_weight, 1.0)):
            weights = {"lap_time": lap_time_weight, "tire_preservation": tire_preservation_weight, "handling_balance": handling_weight}
            with st.spinner("Running Bayesian Optimization..."):
                best_params, predicted_lap = optimize_setup(weights, track_conditions)
            st.success(f"Optimized Lap Time: **{predicted_lap:.3f}s**")
            for key in st.session_state.setup.keys():
                if key in best_params:
                    st.session_state.setup[key] = int(best_params[key])
            st.rerun()
    
    with save_a_col:
        if st.button("Save to Slot A", use_container_width=True):
            st.session_state.setup_A = st.session_state.setup.copy()
            st.toast("✅ Setup A saved!")
    
    with save_b_col:
        if st.button("Save to Slot B", use_container_width=True):
            st.session_state.setup_B = st.session_state.setup.copy()
            st.toast("✅ Setup B saved!")


with col2:
    st.subheader("📊 Live Setup Balance Profile")
    scores = get_balance_scores(st.session_state.setup)
    categories = ['Top Speed', 'Cornering Grip', 'Stability', 'Tire Life']
    fig = go.Figure(data=go.Scatterpolar(r=scores, theta=categories, fill='toself', name='Current Setup', line_color='#00D2BE'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=False, height=350, margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.subheader("⚙️ Current Setup Values")
    st.json(st.session_state.setup)

st.divider()

# --- ENHANCEMENT: Setup Comparison Workbench ---
st.header("⚖️ Setup Comparison Workbench")

if st.session_state.setup_A is None and st.session_state.setup_B is None:
    st.info("Save setups to Slot A and Slot B using the buttons above to compare them side-by-side.")
else:
    comp_col1, comp_col2 = st.columns(2, gap="large")

    with comp_col1:
        st.subheader("Setup A")
        if st.session_state.setup_A:
            scores_A = get_balance_scores(st.session_state.setup_A)
            lap_time_A = get_predicted_lap_time(st.session_state.setup_A, track_conditions)
            st.metric("Predicted Lap Time", f"{lap_time_A:.3f}s")
            
            fig_A = go.Figure(data=go.Scatterpolar(r=scores_A, theta=categories, fill='toself', name='Setup A', line_color='cyan'))
            fig_A.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=False, height=300, margin=dict(t=30, b=30))
            st.plotly_chart(fig_A, use_container_width=True)
            with st.expander("Show Setup A Parameters"):
                st.json(st.session_state.setup_A)
        else:
            st.info("No setup saved in Slot A.")

    with comp_col2:
        st.subheader("Setup B")
        if st.session_state.setup_B:
            scores_B = get_balance_scores(st.session_state.setup_B)
            lap_time_B = get_predicted_lap_time(st.session_state.setup_B, track_conditions)
            
            delta = None
            if st.session_state.setup_A:
                delta = lap_time_B - get_predicted_lap_time(st.session_state.setup_A, track_conditions)

            st.metric("Predicted Lap Time", f"{lap_time_B:.3f}s", delta=f"{delta:.3f}s" if delta is not None else None, delta_color="inverse")

            fig_B = go.Figure(data=go.Scatterpolar(r=scores_B, theta=categories, fill='toself', name='Setup B', line_color='magenta'))
            fig_B.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=False, height=300, margin=dict(t=30, b=30))
            st.plotly_chart(fig_B, use_container_width=True)
            with st.expander("Show Setup B Parameters"):
                st.json(st.session_state.setup_B)
        else:
            st.info("No setup saved in Slot B.")