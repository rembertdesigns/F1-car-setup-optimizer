import streamlit as st
import plotly.graph_objects as go
from optimizer import optimize_setup

st.set_page_config(page_title="F1 Car Setup Optimizer", layout="centered")

st.title("🏎️ F1 Car Setup Optimization")
st.markdown("Optimize for lap time, tire preservation, or handling balance.")

# Sidebar - tradeoff sliders
st.sidebar.header("⚖️ Tradeoff Weights (Sum = 1.0)")

lap_time_weight = st.sidebar.slider("🏁 Lap Time", 0.0, 1.0, 0.6)
tire_preservation_weight = st.sidebar.slider("🛞 Tire Preservation", 0.0, 1.0, 0.2)
handling_weight = st.sidebar.slider("↔️ Handling Balance", 0.0, 1.0, 0.2)

total = lap_time_weight + tire_preservation_weight + handling_weight
if total != 1.0:
    st.sidebar.error("Weights must add up to 1.0")

# Main UI
st.subheader("🎯 Tradeoff-Based Setup Optimization")
if st.button("Run Tradeoff Optimizer") and total == 1.0:
    weights = {
        "lap_time": lap_time_weight,
        "tire_preservation": tire_preservation_weight,
        "handling_balance": handling_weight
    }
    best_params, predicted_lap = optimize_setup(weights)

    st.success(f"🏁 Predicted Lap Time: **{predicted_lap:.2f} sec**")
    st.json(best_params)

    # Radar chart
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
        name='Tradeoff Setup'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        margin=dict(t=30, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)