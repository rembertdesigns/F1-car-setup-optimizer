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


# --- Sidebar: Optimization Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    
    st.subheader("📍 Select Track")
    selected_track = st.selectbox("Choose a track for setup optimization", list(TRACKS_DATA.keys()))
    
    track_info = TRACKS_DATA[selected_track]
    st.image(track_info["image"], use_container_width=True)
    st.info(track_info["description"])

    # Define track conditions based on selection. The optimizer will use these.
    track_conditions = {
        "base_lap_time": track_info.get("base_lap_time", 90.0)
    }

    st.divider()

    st.header("⚖️ Optimization Weights")
    st.markdown("Define what matters most for your optimal setup. Weights must sum to 1.0.")
    
    lap_time_weight = st.slider("🏁 Lap Time Focus", 0.0, 1.0, 0.6, 0.05, help="Prioritize the fastest possible lap time, potentially at the cost of tire wear or stability.")
    tire_preservation_weight = st.slider("🛞 Tire Preservation Focus", 0.0, 1.0, 0.2, 0.05, help="Prioritize a setup that is gentler on the tires, leading to less degradation over a race stint.")
    handling_weight = st.slider("↔️ Handling Balance Focus", 0.0, 1.0, 0.2, 0.05, help="Prioritize a stable and balanced car (e.g., similar front and rear wing angles).")
    
    total_weight = lap_time_weight + tire_preservation_weight + handling_weight
    if not np.isclose(total_weight, 1.0):
        st.sidebar.error(f"Weights must sum to 1.0 (Current sum: {total_weight:.2f})")

# --- Main UI Layout ---
col1, col2 = st.columns([0.6, 0.4], gap="large")

with col1:
    st.subheader("🔧 Car Setup Parameters")
    st.markdown("Adjust the sliders below to create a custom setup, or run the optimizer to find the best one based on your weights.")
    
    if 'setup' not in st.session_state:
        st.session_state.setup = {
            "front_wing_angle": 25, "rear_wing_angle": 25,
            "ride_height": 35, "suspension_stiffness": 5,
            "brake_bias": 54
        }

    setup = st.session_state.setup
    
    # --- ENHANCEMENT: Added help tooltips to sliders ---
    setup["front_wing_angle"] = st.slider("Front Wing Angle", 0, 50, setup["front_wing_angle"], help="Higher angle increases front-end cornering grip but adds drag, reducing top speed.")
    setup["rear_wing_angle"] = st.slider("Rear Wing Angle", 0, 50, setup["rear_wing_angle"], help="Higher angle increases rear-end stability and cornering grip but significantly reduces top speed.")
    setup["ride_height"] = st.slider("Ride Height (mm)", 30, 50, setup["ride_height"], help="Lower ride height generates more downforce from the car's floor but risks 'bottoming out' on bumps.")
    setup["suspension_stiffness"] = st.slider("Suspension Stiffness", 1, 11, setup["suspension_stiffness"], help="Stiffer suspension provides better responsiveness but can increase tire wear and be unstable on bumpy tracks.")
    setup["brake_bias"] = st.slider("Brake Bias (%)", 50, 60, setup["brake_bias"], help="Percentage of brake force sent to the front wheels. Higher values (>55%) increase stability but can cause understeer.")
    
    st.divider()

    st.subheader("🎯 Run Optimization")
    if st.button("Find Optimal Setup", use_container_width=True, type="primary", disabled=not np.isclose(total_weight, 1.0)):
        weights = {
            "lap_time": lap_time_weight,
            "tire_preservation": tire_preservation_weight,
            "handling_balance": handling_weight
        }
        with st.spinner("Running Bayesian Optimization... this may take a moment."):
            best_params, predicted_lap = optimize_setup(weights, track_conditions)
            
        st.success(f"🏁 Optimizer Found Optimal Lap Time: **{predicted_lap:.3f} sec**")
        
        for key in st.session_state.setup.keys():
            if key in best_params:
                st.session_state.setup[key] = int(best_params[key])
        st.rerun()

with col2:
    st.subheader("📊 Setup Balance Profile")
    
    setup_now = st.session_state.setup
    top_speed_score = 10 - ((setup_now["front_wing_angle"] + setup_now["rear_wing_angle"]) / 100) * 10
    cornering_score = ((setup_now["front_wing_angle"] + setup_now["rear_wing_angle"]) / 100) * 8 + (setup_now["suspension_stiffness"] / 11) * 2
    stability_score = ((50 - setup_now["ride_height"]) / 20) * 5 + (setup_now["brake_bias"] / 60) * 5
    tire_life_score = 10 - ((setup_now["suspension_stiffness"] / 11) * 5 + top_speed_score * 0.1)
    
    scores = [np.clip(s, 0, 10) for s in [top_speed_score, cornering_score, stability_score, tire_life_score]]
    categories = ['Top Speed', 'Cornering Grip', 'Stability', 'Tire Life']
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Setup Profile',
        line=dict(color='#00D2BE')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], showticklabels=False, ticks=''),
            angularaxis=dict(tickfont=dict(size=14))
        ),
        showlegend=False,
        height=350,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- ENHANCEMENT: Added expander with explanation for the chart ---
    with st.expander("💡 How to read the Setup Balance Profile"):
        st.markdown("""
            This radar chart provides a visual summary of the performance trade-offs for the current setup. A larger area generally indicates a more capable setup.
            
            - **Top Speed:** Favored by **low** front and rear wing angles.
            - **Cornering Grip:** Favored by **high** front and rear wing angles.
            - **Stability:** Higher values are achieved with a **lower** ride height and a more **balanced** brake bias (closer to 55%).
            - **Tire Life:** Higher for **softer** (lower) suspension stiffness settings.
        """)

    st.divider()

    st.subheader("⚙️ Current / Optimal Setup Values")
    st.json(st.session_state.setup)