import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import os
import json  # ✅ for exporting to JSON
from fpdf import FPDF  # ✅ for PDF generation
from fpdf.enums import XPos, YPos  # ✅ for PDF layout control
from optimizer import optimize_setup, run_pareto_optimization

st.set_page_config(page_title="F1 Car Setup Optimizer", layout="wide")

st.title("🏎️ F1 Car Setup Workbench")
st.markdown("Interactively create and optimize a car setup for different performance tradeoffs.")
st.divider()

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
    base_lap = track_conditions.get('base_lap_time', 90.0)
    aero_penalty = (setup_dict["front_wing_angle"] + setup_dict["rear_wing_angle"]) * 0.04
    cornering_bonus = (setup_dict["front_wing_angle"] + setup_dict["rear_wing_angle"]) * 0.02
    ride_height_effect = (setup_dict["ride_height"] - 30) * 0.05
    return base_lap + aero_penalty - cornering_bonus + ride_height_effect

def generate_telemetry_trace(setup_dict):
    """Generates a mock speed vs. distance trace for a generic lap."""
    total_distance = 5000; distance = np.linspace(0, total_distance, 500)
    straight1, corner1, straight2, corner2, straight3 = 1000, 1500, 3000, 3500, total_distance
    max_s = 350 - (setup_dict["front_wing_angle"] + setup_dict["rear_wing_angle"]) * 0.5
    min_s = 80 + (setup_dict["front_wing_angle"] + setup_dict["rear_wing_angle"]) * 0.3
    fast_s = 200 + (setup_dict["front_wing_angle"] + setup_dict["rear_wing_angle"]) * 0.4
    speed = np.zeros_like(distance)
    speed[distance <= straight1] = max_s; speed[(distance > straight1) & (distance <= corner1)] = min_s
    speed[(distance > corner1) & (distance <= straight2)] = max_s; speed[(distance > straight2) & (distance <= corner2)] = fast_s
    speed[distance > corner2] = max_s
    speed = np.convolve(speed, np.ones(15)/15, mode='same')
    return distance, speed

# --- NEW: PDF Generation for a single setup ---
def generate_setup_pdf(setup_dict, track_name, lap_time):
    """Creates a PDF summary of a car setup."""
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

# --- Initialize Session State ---
if 'setup' not in st.session_state:
    st.session_state.setup = {"front_wing_angle": 25, "rear_wing_angle": 25, "ride_height": 35, "suspension_stiffness": 5, "brake_bias": 54}
if 'setup_A' not in st.session_state:
    st.session_state.setup_A = None
if 'setup_B' not in st.session_state:
    st.session_state.setup_B = None
if 'pareto_front' not in st.session_state:
    st.session_state.pareto_front = None

# --- NEW: Handle Shareable Links (URL Query Params) ---
# This runs once at the beginning of the script execution
if not st.session_state.get('url_params_loaded', False):
    query_params = st.query_params
    try:
        if 'fwa' in query_params: st.session_state.setup['front_wing_angle'] = int(query_params['fwa'][0])
        if 'rwa' in query_params: st.session_state.setup['rear_wing_angle'] = int(query_params['rwa'][0])
        if 'rh' in query_params: st.session_state.setup['ride_height'] = int(query_params['rh'][0])
        if 'ss' in query_params: st.session_state.setup['suspension_stiffness'] = int(query_params['ss'][0])
        if 'bb' in query_params: st.session_state.setup['brake_bias'] = int(query_params['bb'][0])
    except (ValueError, IndexError):
        st.toast("⚠️ Could not parse setup from URL. Using default.", icon="⚠️")
    st.session_state.url_params_loaded = True

# --- App Layout ---
main_cols = st.columns([0.25, 0.4, 0.35], gap="large")

# --- COLUMN 1: CONTROLS (SIDEBAR-LIKE) ---
with main_cols[0]:
    st.header("⚙️ Controls")
    st.subheader("📍 Select Track & Conditions")
    selected_track = st.selectbox("Track", list(TRACKS_DATA.keys()), label_visibility="collapsed")
    track_info = TRACKS_DATA[selected_track]
    st.image(track_info["image"], use_container_width=True)
    st.info(track_info["description"])
    
    # Define track conditions based on selection. The optimizer will use these.
    track_conditions = {
        "base_lap_time": track_info.get("base_lap_time", 90.0),
        "track_temperature": st.slider("Track Temperature (°C)", 15.0, 50.0, track_info.get("track_temperature", 30.0)),
        "grip_level": st.slider("Grip Level", 0.8, 1.2, track_info.get("grip_level", 1.0), 0.01),
        "tire_compound": st.selectbox("Target Tire Compound", ["soft", "medium", "hard"], index=0)
    }

    st.divider()
    st.header("⚖️ Optimization Weights")
    st.caption("Define what matters most for the single-setup optimizer.")
    lap_time_weight = st.slider("🏁 Lap Time Focus", 0.0, 1.0, 0.6, 0.05)
    tire_preservation_weight = st.slider("🛞 Tire Preservation", 0.0, 1.0, 0.2, 0.05)
    handling_weight = st.slider("↔️ Handling Balance", 0.0, 1.0, 0.2, 0.05)
    total_weight = lap_time_weight + tire_preservation_weight + handling_weight
    if not np.isclose(total_weight, 1.0):
        st.error(f"Weights must sum to 1.0 (is {total_weight:.2f})")

# --- COLUMN 2: SETUP WORKBENCH ---
with main_cols[1]:
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

if c1.button("Find Optimal Setup", use_container_width=True, type="primary", disabled=not np.isclose(total_weight, 1.0)):
    weights = {
        "lap_time": lap_time_weight,
        "tire_preservation": tire_preservation_weight,
        "handling_balance": handling_weight
    }
    with st.spinner("Running Bayesian Optimization..."):
        best_params, predicted_lap = optimize_setup(weights, track_conditions)

    st.success(f"Optimized Lap Time: {predicted_lap:.3f}s")
    for k, v in best_params.items():
        st.session_state.setup[k] = int(v)
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


# --- Sensitivity Analysis Section ---
st.divider()
st.header("🔍 Setup Sensitivity Analysis")
if 'last_optimal_setup' not in st.session_state:
    st.session_state.last_optimal_setup = None

if st.session_state.setup:
    st.session_state.last_optimal_setup = st.session_state.setup.copy()

if st.session_state.last_optimal_setup:
    st.subheader("Analyze Setup Parameter Sensitivity")
    param_to_test = st.selectbox("Choose parameter to analyze:", [
        "front_wing_angle", "rear_wing_angle", "ride_height", "suspension_stiffness", "brake_bias"])
    range_val = st.slider("Change range (+/- value):", 1, 10, 5)
    if st.button("Run Sensitivity Analysis", use_container_width=True):
        param_range = range(st.session_state.last_optimal_setup[param_to_test] - range_val,
                            st.session_state.last_optimal_setup[param_to_test] + range_val + 1)
        lap_times = []
        x_vals = []

        for val in param_range:
            test_setup = st.session_state.last_optimal_setup.copy()
            test_setup[param_to_test] = val
            lap_time = get_predicted_lap_time(test_setup, track_conditions)
            x_vals.append(val)
            lap_times.append(lap_time)

        fig_sensitivity = go.Figure()
        fig_sensitivity.add_trace(go.Scatter(x=x_vals, y=lap_times, mode='lines+markers', name='Lap Time vs. Value'))
        fig_sensitivity.update_layout(
            title=f"Sensitivity of Lap Time to {param_to_test.replace('_',' ').title()}",
            xaxis_title=f"{param_to_test.replace('_',' ').title()}",
            yaxis_title="Predicted Lap Time (s)",
            height=350
        )
        st.plotly_chart(fig_sensitivity, use_container_width=True)
else:
    st.info("Run an optimization first to analyze sensitivity around an optimal setup.")

# --- COLUMN 3: LIVE ANALYSIS ---
with main_cols[2]:
    st.header("📊 Live Analysis")
    st.subheader("Setup Balance Profile")
    scores = get_balance_scores(st.session_state.setup)
    categories = ['Top Speed', 'Cornering Grip', 'Stability', 'Tire Life']
    fig = go.Figure(data=go.Scatterpolar(r=scores, theta=categories, fill='toself', name='Current Setup', line_color='#00D2BE'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=False, height=300, margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🛰️ Live Telemetry Trace")
    distance, speed = generate_telemetry_trace(st.session_state.setup)
    fig_telemetry = go.Figure(data=go.Scatter(x=distance, y=speed, mode='lines', name='Speed', line=dict(color='#FF8700', width=4)))
    fig_telemetry.update_layout(title="Simulated Speed Trace", xaxis_title="Distance (m)", yaxis_title="Speed (km/h)", height=250, margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig_telemetry, use_container_width=True)

# --- Pareto Front Optimization & Comparison Section ---
st.divider()
st.header("📈 Pareto Analysis: Lap Time vs. Tire Preservation")
st.markdown("Find a range of optimal setups that represent the best possible trade-offs between pure speed and tire life. Click on any point to see its setup parameters.")

if st.button("Find Optimal Trade-offs (Pareto Front)"):
    with st.spinner("Running Multi-Objective Optimization... This will take longer."):
        pareto_df = run_pareto_optimization(track_conditions, n_steps=15)
        st.session_state.pareto_front = pareto_df

if st.session_state.pareto_front is not None:
    df_pareto = st.session_state.pareto_front
    
    hover_text = [
        f"<b>Lap Time: {row.lap_time:.3f}s</b><br>Tire Score: {row.tire_preservation:.2f}<br>---<br>" +
        "<br>".join([f"<b>{k.replace('_', ' ').title()}:</b> {v}" for k, v in row.setup.items()])
        for row in df_pareto.itertuples()
    ]

    fig_pareto = px.scatter(
        df_pareto, 
        x="lap_time", y="tire_preservation", 
        title="Optimal Setups: Lap Time vs. Tire Preservation Trade-off",
        labels={"lap_time": "Predicted Lap Time (s) - Lower is Better →", "tire_preservation": "Tire Preservation Score - Higher is Better →"},
        color="tire_preservation",
        color_continuous_scale=px.colors.sequential.Viridis,
        hover_name=hover_text
    )
    fig_pareto.update_traces(customdata=hover_text, hovertemplate='%{customdata}<extra></extra>', marker=dict(size=12, line=dict(width=1,color='DarkSlateGrey')))
    st.plotly_chart(fig_pareto, use_container_width=True)

# --- Setup Comparison Section ---
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
            with st.expander("Show Setup A Parameters"): st.json(st.session_state.setup_A)
        else: st.info("No setup in Slot A.")
    with c2:
        st.subheader("Setup B")
        if st.session_state.setup_B:
            lap_time_B = get_predicted_lap_time(st.session_state.setup_B, track_conditions)
            delta = lap_time_B - get_predicted_lap_time(st.session_state.setup_A, track_conditions) if st.session_state.setup_A else None
            st.metric("Predicted Lap Time", f"{lap_time_B:.3f}s", delta=f"{delta:.3f}s" if delta is not None else None, delta_color="inverse")
            with st.expander("Show Setup B Parameters"): st.json(st.session_state.setup_B)
        else: st.info("No setup in Slot B.")

    if st.session_state.setup_A and st.session_state.setup_B:
        st.subheader("🛰️ Comparative Telemetry & Balance Overlay")
        
        fig_comp_tele = go.Figure()
        dist_A, speed_A = generate_telemetry_trace(st.session_state.setup_A); fig_comp_tele.add_trace(go.Scatter(x=dist_A, y=speed_A, mode='lines', name='Setup A', line=dict(color='cyan', width=4)))
        dist_B, speed_B = generate_telemetry_trace(st.session_state.setup_B); fig_comp_tele.add_trace(go.Scatter(x=dist_B, y=speed_B, mode='lines', name='Setup B', line=dict(color='magenta', width=4, dash='dot')))
        fig_comp_tele.update_layout(title="Speed Trace Comparison", xaxis_title="Distance (m)", yaxis_title="Speed (km/h)", legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
        
        fig_comp_radar = go.Figure()
        categories = ['Top Speed', 'Cornering Grip', 'Stability', 'Tire Life']
        scores_A = get_balance_scores(st.session_state.setup_A); fig_comp_radar.add_trace(go.Scatterpolar(r=scores_A, theta=categories, fill='toself', name='Setup A', line_color='cyan', opacity=0.7))
        scores_B = get_balance_scores(st.session_state.setup_B); fig_comp_radar.add_trace(go.Scatterpolar(r=scores_B, theta=categories, fill='toself', name='Setup B', line_color='magenta', opacity=0.7))
        fig_comp_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), title="Setup Balance Comparison", legend=dict(yanchor="bottom",y=0,xanchor="left",x=0))

        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(fig_comp_tele, use_container_width=True)
        with c2: st.plotly_chart(fig_comp_radar, use_container_width=True)

# --- NEW: Export and Sharing Section ---
st.divider()
st.header("📤 Export & Share")
st.markdown("Download the current setup from the workbench or generate a shareable link.")

current_setup = st.session_state.setup
current_lap_time = get_predicted_lap_time(current_setup, track_conditions)

export_cols = st.columns(3)
with export_cols[0]:
    # JSON Export
    json_string = json.dumps(current_setup, indent=4)
    st.download_button(
        label="Download as JSON",
        file_name=f"f1_setup_{selected_track}.json",
        mime="application/json",
        data=json_string,
        use_container_width=True
    )
with export_cols[1]:
    # PDF Export
    pdf_bytes = generate_setup_pdf(current_setup, selected_track, current_lap_time)
    st.download_button(
        label="Download as PDF",
        file_name=f"f1_setup_{selected_track}.pdf",
        mime="application/pdf",
        data=pdf_bytes,
        use_container_width=True
    )

# Shareable Link Generation
base_url = "https://your-app-url.streamlit.app/" # Replace with your deployed app's URL
query_params = f"?fwa={setup['front_wing_angle']}&rwa={setup['rear_wing_angle']}&rh={setup['ride_height']}&ss={setup['suspension_stiffness']}&bb={setup['brake_bias']}"
share_url = base_url + query_params

with export_cols[2]:
    if st.button("Generate Shareable Link", use_container_width=True):
        st.code(share_url, language=None)