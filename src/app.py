import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
from optimizer import optimize_setup

st.set_page_config(page_title="F1 Car Setup Optimizer", layout="centered")

st.title("🏎️ F1 Car Setup Optimization")
st.markdown("Get the best setup for performance based on your conditions.")

# --- Sidebar for user-defined conditions (overrides optimizer default if needed)
st.sidebar.header("📋 Track Conditions")
user_temp = st.sidebar.slider("Track Temperature (°C)", 15, 45, 30)
user_grip = st.sidebar.slider("Grip Level", 0.8, 1.2, 1.0)

# --- Run optimization
st.subheader("🔍 Optimized Setup")
if st.button("Optimize Setup"):
    best_params, predicted_lap = optimize_setup()

    # Show results
    st.success(f"Predicted Lap Time: **{predicted_lap:.2f} sec**")

    st.json(best_params)

    # --- Radar chart of setup parameters
    radar_features = [
        "front_wing_angle", "rear_wing_angle", "ride_height",
        "suspension_stiffness", "brake_bias"
    ]
    radar_values = [best_params[f] for f in radar_features]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_values + [radar_values[0]],
        theta=radar_features + [radar_features[0]],
        fill='toself',
        name='Optimized Setup'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        margin=dict(t=30, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)