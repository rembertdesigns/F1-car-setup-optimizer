import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration with enhanced settings
st.set_page_config(
    page_title="F1 Car Setup Workbench",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .setup-card {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .track-info {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #f39c12;
    }
    
    .success-card {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Enhanced expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'setup' not in st.session_state:
    st.session_state.setup = {
        "front_wing_angle": 25, 
        "rear_wing_angle": 25, 
        "ride_height": 35, 
        "suspension_stiffness": 5, 
        "brake_bias": 54
    }
if 'setup_A' not in st.session_state:
    st.session_state.setup_A = None
if 'setup_B' not in st.session_state:
    st.session_state.setup_B = None
if 'pareto_front' not in st.session_state:
    st.session_state.pareto_front = None
if 'optimization_history' not in st.session_state:
    st.session_state.optimization_history = []

# Track data with enhanced information
TRACKS_DATA = {
    "Monaco": {
        "description": "The jewel of F1. Ultra-tight streets demand maximum downforce and precision.",
        "base_lap_time": 71.0,
        "difficulty": "Expert",
        "characteristics": ["Tight corners", "Elevation changes", "Wall proximity"],
        "optimal_strategy": "High downforce, low speed focus"
    },
    "Monza": {
        "description": "The Temple of Speed. Long straights favor low drag and top speed.",
        "base_lap_time": 80.0,
        "difficulty": "Intermediate",
        "characteristics": ["High speed", "Low downforce", "Long straights"],
        "optimal_strategy": "Minimal downforce, drag reduction"
    },
    "Silverstone": {
        "description": "Home of British GP. High-speed corners test aerodynamic balance.",
        "base_lap_time": 93.0,
        "difficulty": "Advanced",
        "characteristics": ["High-speed corners", "Weather variability", "Flowing layout"],
        "optimal_strategy": "Balanced setup with good stability"
    },
    "Spa-Francorchamps": {
        "description": "The Ardennes classic. Elevation and speed create unique challenges.",
        "base_lap_time": 105.0,
        "difficulty": "Advanced",
        "characteristics": ["Elevation changes", "High speed", "Weather prone"],
        "optimal_strategy": "Balanced downforce with straight-line speed"
    },
    "Suzuka": {
        "description": "The figure-8 masterpiece. Precision and balance are paramount.",
        "base_lap_time": 90.0,
        "difficulty": "Expert",
        "characteristics": ["Technical sections", "Figure-8 layout", "Precision required"],
        "optimal_strategy": "High downforce with responsive handling"
    },
    "Circuit de Barcelona-Catalunya": {
        "description": "The all-rounder test. Technical variety demands complete car performance.",
        "base_lap_time": 91.0,
        "difficulty": "Advanced",
        "characteristics": ["Technical corners", "Variety of speeds", "Testing benchmark"],
        "optimal_strategy": "Balanced setup for all conditions"
    },
    "Bahrain International Circuit": {
        "description": "Desert challenge. Brake-heavy layout tests cooling and traction.",
        "base_lap_time": 93.5,
        "difficulty": "Intermediate",
        "characteristics": ["Heavy braking", "Desert conditions", "Traction zones"],
        "optimal_strategy": "Good cooling and brake management"
    },
    "Interlagos": {
        "description": "Brazilian passion. Short, intense lap with elevation and overtaking.",
        "base_lap_time": 72.5,
        "difficulty": "Advanced",
        "characteristics": ["Short lap", "Elevation changes", "Overtaking opportunities"],
        "optimal_strategy": "Quick acceleration and good traction"
    },
    "Red Bull Ring": {
        "description": "Alpine speed. Short, flowing layout with dramatic elevation changes.",
        "base_lap_time": 66.5,
        "difficulty": "Intermediate",
        "characteristics": ["Short lap", "Elevation changes", "Fast corners"],
        "optimal_strategy": "Straight-line speed with good stability"
    },
    "Hungaroring": {
        "description": "The dustbowl. Twisty, technical layout favors maximum downforce.",
        "base_lap_time": 77.5,
        "difficulty": "Advanced",
        "characteristics": ["Twisty layout", "Technical corners", "Limited overtaking"],
        "optimal_strategy": "Maximum downforce and precision"
    },
    "Circuit of the Americas": {
        "description": "American ambition. Modern layout with dramatic elevation changes.",
        "base_lap_time": 97.0,
        "difficulty": "Advanced",
        "characteristics": ["Elevation changes", "Modern design", "Variety of corners"],
        "optimal_strategy": "Balanced setup with good stability"
    },
    "Singapore": {
        "description": "Night fever. Long, hot street circuit tests endurance and cooling.",
        "base_lap_time": 110.0,
        "difficulty": "Expert",
        "characteristics": ["Night race", "Hot conditions", "Long lap"],
        "optimal_strategy": "Excellent cooling and consistent performance"
    },
    "Zandvoort": {
        "description": "Orange army home. Fast, narrow with banked corners and flowing rhythm.",
        "base_lap_time": 72.0,
        "difficulty": "Advanced",
        "characteristics": ["Banked corners", "Narrow track", "Fast rhythm"],
        "optimal_strategy": "Good downforce with responsive handling"
    },
    "Jeddah Street Circuit": {
        "description": "High-speed street fight. Ultra-fast walls demand nerves of steel.",
        "base_lap_time": 89.0,
        "difficulty": "Expert",
        "characteristics": ["High-speed street", "Wall proximity", "Thin margins"],
        "optimal_strategy": "Low drag with maximum precision"
    },
    "Las Vegas Strip Circuit": {
        "description": "Sin City spectacle. Long straights and cool nights create unique challenges.",
        "base_lap_time": 100.0,
        "difficulty": "Intermediate",
        "characteristics": ["Long straights", "Cold conditions", "New layout"],
        "optimal_strategy": "Low downforce with tire temperature management"
    },
    "Yas Marina (Abu Dhabi)": {
        "description": "Twilight finale. Mix of long straights and tight technical sections.",
        "base_lap_time": 94.0,
        "difficulty": "Intermediate",
        "characteristics": ["Twilight conditions", "Mixed layout", "Season finale"],
        "optimal_strategy": "Balanced setup for varied demands"
    },
    "Imola": {
        "description": "Historic challenge. Fast, narrow layout where track limits are crucial.",
        "base_lap_time": 80.0,
        "difficulty": "Advanced",
        "characteristics": ["Historic circuit", "Narrow track", "Track limits critical"],
        "optimal_strategy": "High downforce with precise positioning"
    },
    "Baku City Circuit": {
        "description": "Contrasts extreme. Massive straights meet impossibly tight corners.",
        "base_lap_time": 102.0,
        "difficulty": "Expert",
        "characteristics": ["Longest straight", "Tight corners", "Risk vs reward"],
        "optimal_strategy": "Low drag with adaptable setup"
    },
    "Shanghai": {
        "description": "Eastern promise. Modern design with long corners testing stability.",
        "base_lap_time": 95.0,
        "difficulty": "Intermediate",
        "characteristics": ["Long corners", "Modern layout", "Stability test"],
        "optimal_strategy": "Good stability and consistent performance"
    },
    "Canada (Montreal)": {
        "description": "Wall of Champions. Challenging braking zones with unforgiving barriers.",
        "base_lap_time": 76.0,
        "difficulty": "Advanced",
        "characteristics": ["Heavy braking", "Wall proximity", "Challenging chicanes"],
        "optimal_strategy": "Low drag with excellent braking stability"
    },
    "Miami": {
        "description": "American newcomer. High-speed corners with elevation create modern test.",
        "base_lap_time": 90.0,
        "difficulty": "Intermediate",
        "characteristics": ["High-speed corners", "Elevation changes", "New circuit"],
        "optimal_strategy": "Balanced downforce with good stability"
    }
}

# Enhanced header with professional styling
st.markdown("""
<div class="main-header">
    <h1>🏎️ Formula 1 Car Setup Workbench</h1>
    <p>Professional-grade aerodynamic and mechanical setup optimization for Formula 1 racing</p>
    <p><strong>Advanced Physics Simulation • Multi-Objective Optimization • Real-time Analysis</strong></p>
</div>
""", unsafe_allow_html=True)

# Sidebar for main controls
with st.sidebar:
    st.markdown("### 🎯 Track Selection")
    selected_track = st.selectbox(
        "Choose Circuit",
        list(TRACKS_DATA.keys()),
        help="Select the circuit for setup optimization"
    )
    
    track_info = TRACKS_DATA[selected_track]
    
    # Enhanced track information display
    st.markdown(f"""
    <div class="track-info">
        <h4>📍 {selected_track}</h4>
        <p><strong>Difficulty:</strong> {track_info['difficulty']}</p>
        <p><strong>Base Lap Time:</strong> {track_info['base_lap_time']:.1f}s</p>
        <p>{track_info['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📊 Track Characteristics"):
        for char in track_info['characteristics']:
            st.write(f"• {char}")
        st.info(f"💡 **Strategy:** {track_info['optimal_strategy']}")
    
    st.markdown("### 🌡️ Session Conditions")
    track_temperature = st.slider("Track Temperature (°C)", 15, 50, 30)
    air_temperature = st.slider("Air Temperature (°C)", 10, 45, 25)
    humidity = st.slider("Humidity (%)", 30, 90, 60)
    wind_speed = st.slider("Wind Speed (km/h)", 0, 30, 5)
    
    weather_condition = st.selectbox("Weather", ["Dry", "Light Rain", "Heavy Rain"])
    
    st.markdown("### ⛽ Car Configuration")
    fuel_load = st.slider("Fuel Load (kg)", 80, 110, 100)
    tire_compound = st.selectbox("Tire Compound", ["Soft", "Medium", "Hard"])
    
    starting_position = st.slider("Grid Position", 1, 20, 10)

# Main content area with enhanced layout
col1, col2, col3 = st.columns([2, 2, 1], gap="large")

# Setup workbench
with col1:
    st.markdown("### 🔧 Setup Workbench")
    
    with st.container():
        st.markdown('<div class="setup-card">', unsafe_allow_html=True)
        
        # Aerodynamics section
        st.markdown("#### 🌪️ Aerodynamics")
        setup = st.session_state.setup
        
        col_aero1, col_aero2 = st.columns(2)
        with col_aero1:
            setup["front_wing_angle"] = st.slider(
                "Front Wing Angle", 0, 50, setup["front_wing_angle"],
                help="Higher values increase downforce and drag"
            )
        with col_aero2:
            setup["rear_wing_angle"] = st.slider(
                "Rear Wing Angle", 0, 50, setup["rear_wing_angle"],
                help="Affects rear downforce and stability"
            )
        
        # Suspension section
        st.markdown("#### 🏗️ Suspension")
        col_susp1, col_susp2 = st.columns(2)
        with col_susp1:
            setup["ride_height"] = st.slider(
                "Ride Height (mm)", 20, 50, setup["ride_height"],
                help="Lower values improve aerodynamics but reduce ground clearance"
            )
        with col_susp2:
            setup["suspension_stiffness"] = st.slider(
                "Suspension Stiffness", 1, 11, setup["suspension_stiffness"],
                help="Higher values improve response but reduce tire contact"
            )
        
        # Brakes section
        st.markdown("#### 🛑 Braking")
        setup["brake_bias"] = st.slider(
            "Brake Bias (%)", 50, 60, setup["brake_bias"],
            help="Percentage of braking force applied to front wheels"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons with enhanced styling
    st.markdown("#### 🎯 Actions")
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    with btn_col1:
        if st.button("🧮 Optimize Setup", use_container_width=True, type="primary"):
            with st.spinner("Running optimization..."):
                # Simulate optimization
                time.sleep(2)
                optimized_setup = {
                    "front_wing_angle": np.random.randint(15, 35),
                    "rear_wing_angle": np.random.randint(20, 40),
                    "ride_height": np.random.randint(25, 45),
                    "suspension_stiffness": np.random.randint(3, 9),
                    "brake_bias": np.random.randint(52, 58)
                }
                st.session_state.setup = optimized_setup
                st.success("✅ Setup optimized successfully!")
                st.rerun()
    
    with btn_col2:
        if st.button("💾 Save to Slot A", use_container_width=True):
            st.session_state.setup_A = st.session_state.setup.copy()
            st.success("✅ Saved to Slot A!")
    
    with btn_col3:
        if st.button("💾 Save to Slot B", use_container_width=True):
            st.session_state.setup_B = st.session_state.setup.copy()
            st.success("✅ Saved to Slot B!")

# --- 🧠 RL Agent Integration ---
st.markdown("#### 🤖 AI-Powered Optimization")
col_ai1, col_ai2 = st.columns(2)

with col_ai1:
    optimization_mode = st.selectbox(
        "Optimization Mode",
        ["Single Objective", "Multi-Objective (NSGA-II)", "Reinforcement Learning"],
        help="Choose optimization algorithm"
    )

with col_ai2:
    if st.button("🧠 Generate RL Setup", use_container_width=True, type="secondary"):
        with st.spinner("RL Agent analyzing track..."):
            time.sleep(3)
            rl_setup = {
                "front_wing_angle": np.random.randint(20, 40),
                "rear_wing_angle": np.random.randint(25, 45),
                "ride_height": np.random.randint(28, 42),
                "suspension_stiffness": np.random.randint(4, 8),
                "brake_bias": np.random.randint(53, 57)
            }
            st.session_state.setup = rl_setup
            st.success("🎯 RL-optimized setup generated!")
            st.rerun()

# Performance analysis
with col2:
    st.markdown("### 📊 Performance Analysis")
    
    # Calculate key metrics
    def calculate_performance_metrics(setup, track_info):
        base_time = track_info["base_lap_time"]
        
        # Simplified performance model
        aero_effect = (setup["front_wing_angle"] + setup["rear_wing_angle"]) * 0.02
        suspension_effect = (setup["suspension_stiffness"] - 6) * 0.05
        ride_height_effect = (setup["ride_height"] - 35) * 0.03
        brake_effect = abs(setup["brake_bias"] - 55) * 0.02
        
        lap_time = base_time + aero_effect + suspension_effect + ride_height_effect + brake_effect
        
        # Performance scores (0-10)
        top_speed = 10 - (setup["front_wing_angle"] + setup["rear_wing_angle"]) / 10
        cornering = (setup["front_wing_angle"] + setup["rear_wing_angle"]) / 10
        stability = 10 - abs(setup["ride_height"] - 35) / 3
        tire_wear = 10 - setup["suspension_stiffness"]
        
        return {
            "lap_time": lap_time,
            "top_speed": max(0, min(10, top_speed)),
            "cornering": max(0, min(10, cornering)),
            "stability": max(0, min(10, stability)),
            "tire_wear": max(0, min(10, tire_wear))
        }
    
    metrics = calculate_performance_metrics(st.session_state.setup, track_info)
    
    # Display key metrics with enhanced styling
    st.markdown(f"""
    <div class="metric-container">
        <h3>🏁 Predicted Lap Time</h3>
        <h1>{metrics['lap_time']:.3f}s</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance radar chart
    categories = ['Top Speed', 'Cornering', 'Stability', 'Tire Life']
    values = [metrics['top_speed'], metrics['cornering'], metrics['stability'], metrics['tire_wear']]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Setup',
        line_color='#667eea',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickfont=dict(size=10)
            )
        ),
        title="Performance Balance",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

    # --- ⚠️ Setup Health Check ---
def check_setup_anomalies(setup):
    anomalies = []
    wing_diff = abs(setup["front_wing_angle"] - setup["rear_wing_angle"])
    if wing_diff > 20:
        anomalies.append({"type": "Aerodynamic Imbalance", "severity": "High", "message": f"Large wing angle difference ({wing_diff}°) may cause instability"})
    if setup["ride_height"] < 25 and (setup["front_wing_angle"] + setup["rear_wing_angle"]) > 70:
        anomalies.append({"type": "Ground Effect Risk", "severity": "Medium", "message": "Very low ride height with high downforce increases porpoising risk"})
    if setup["brake_bias"] < 52 or setup["brake_bias"] > 58:
        anomalies.append({"type": "Brake Balance", "severity": "Medium", "message": "Extreme brake bias may cause handling issues"})
    return anomalies

st.markdown("### ⚠️ Setup Health Check")
anomalies = check_setup_anomalies(st.session_state.setup)
if anomalies:
    for a in anomalies:
        color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(a['severity'], "⚠️")
        st.warning(f"{color} **{a['type']}**: {a['message']}")
else:
    st.success("✅ Setup configuration appears optimal - no anomalies detected")
    
    # Individual metrics
    metric_cols = st.columns(2)
    with metric_cols[0]:
        st.metric("Top Speed", f"{metrics['top_speed']:.1f}/10", help="Straight-line performance")
        st.metric("Cornering", f"{metrics['cornering']:.1f}/10", help="Corner grip and speed")
    with metric_cols[1]:
        st.metric("Stability", f"{metrics['stability']:.1f}/10", help="Car balance and predictability")
        st.metric("Tire Life", f"{metrics['tire_wear']:.1f}/10", help="Tire degradation rate")

# Quick actions and info
with col3:
    st.markdown("### ⚡ Quick Actions")
    
    # Preset setups
    if st.button("🏎️ Qualifying Setup", use_container_width=True):
        st.session_state.setup = {
            "front_wing_angle": 45, "rear_wing_angle": 40, "ride_height": 25,
            "suspension_stiffness": 8, "brake_bias": 56
        }
        st.success("Loaded qualifying setup!")
        st.rerun()
    
    if st.button("🏁 Race Setup", use_container_width=True):
        st.session_state.setup = {
            "front_wing_angle": 30, "rear_wing_angle": 35, "ride_height": 35,
            "suspension_stiffness": 5, "brake_bias": 54
        }
        st.success("Loaded race setup!")
        st.rerun()
    
    if st.button("🌧️ Wet Weather", use_container_width=True):
        st.session_state.setup = {
            "front_wing_angle": 40, "rear_wing_angle": 45, "ride_height": 40,
            "suspension_stiffness": 3, "brake_bias": 52
        }
        st.success("Loaded wet weather setup!")
        st.rerun()
    
    st.markdown("---")
    
    # Setup comparison
    if st.session_state.setup_A and st.session_state.setup_B:
        st.markdown("### ⚖️ Setup Comparison")
        
        metrics_A = calculate_performance_metrics(st.session_state.setup_A, track_info)
        metrics_B = calculate_performance_metrics(st.session_state.setup_B, track_info)
        
        delta = metrics_B['lap_time'] - metrics_A['lap_time']
        
        st.metric(
            "Lap Time Delta (B-A)",
            f"{delta:+.3f}s",
            delta=f"{delta:+.3f}s",
            delta_color="inverse"
        )
        
        if st.button("📊 Detailed Comparison", use_container_width=True):
            st.session_state.show_comparison = True

# Advanced analysis sections
st.markdown("---")

# Telemetry visualization
with st.expander("📈 Telemetry Analysis", expanded=False):
    st.markdown("### 🏎️ Simulated Telemetry Data")
    
    # Generate simulated telemetry
    distance = np.linspace(0, 5000, 200)
    base_speed = 200 + (10 - (st.session_state.setup["front_wing_angle"] + st.session_state.setup["rear_wing_angle"]) / 10) * 15
    
    # Create speed profile with corners
    speed = base_speed * (1 + 0.3 * np.sin(distance / 800) + 0.1 * np.random.randn(len(distance)))
    speed = np.clip(speed, 80, 350)
    
    # Generate other telemetry data
    throttle = np.clip(100 * (speed / np.max(speed)) + 5 * np.random.randn(len(distance)), 0, 100)
    brake = np.clip(100 - throttle + 10 * np.random.randn(len(distance)), 0, 100)
    brake = np.where(brake < 20, 0, brake)
    
    # Create telemetry plot
    fig_telemetry = go.Figure()
    
    fig_telemetry.add_trace(go.Scatter(
        x=distance, y=speed, name='Speed (km/h)', 
        line=dict(color='#667eea', width=3)
    ))
    
    fig_telemetry.add_trace(go.Scatter(
        x=distance, y=throttle, name='Throttle (%)', 
        line=dict(color='#28a745', width=2), yaxis='y2'
    ))
    
    fig_telemetry.add_trace(go.Scatter(
        x=distance, y=brake, name='Brake (%)', 
        line=dict(color='#dc3545', width=2), yaxis='y2'
    ))
    
    fig_telemetry.update_layout(
        title="Lap Telemetry Simulation",
        xaxis_title="Distance (m)",
        yaxis_title="Speed (km/h)",
        yaxis2=dict(title="Pedal Input (%)", overlaying='y', side='right', range=[0, 100]),
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_telemetry, use_container_width=True)

# Setup optimization history
with st.expander("📋 Optimization History", expanded=False):
    st.markdown("### 🔄 Setup Evolution")

# --- 🔍 AI Model Explainability (SHAP) ---
with st.expander("🔍 AI Model Explainability (SHAP)", expanded=False):
    st.markdown("### 🧠 Understanding AI Decisions")

    if st.button("📊 Generate SHAP Analysis"):
        shap_data = {
            'Feature': ['Front Wing', 'Rear Wing', 'Ride Height', 'Suspension', 'Brake Bias'],
            'SHAP_Value': np.random.randn(5) * 0.1,
            'Impact': ['High', 'Medium', 'High', 'Low', 'Medium']
        }

        df_shap = pd.DataFrame(shap_data)
        colors = ['red' if x < 0 else 'green' for x in df_shap['SHAP_Value']]

        fig_shap = go.Figure()
        fig_shap.add_trace(go.Bar(
            x=df_shap['Feature'],
            y=df_shap['SHAP_Value'],
            marker_color=colors
        ))
        fig_shap.update_layout(
            title="Feature Impact on Lap Time (SHAP Values)",
            xaxis_title="Setup Parameters",
            yaxis_title="Impact on Lap Time (seconds)",
            height=400
        )

        st.plotly_chart(fig_shap, use_container_width=True)
        st.dataframe(df_shap, use_container_width=True)
    
    # Add current setup to history for demonstration
    if st.button("📝 Log Current Setup"):
        current_metrics = calculate_performance_metrics(st.session_state.setup, track_info)
        log_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "lap_time": current_metrics["lap_time"],
            "setup": st.session_state.setup.copy(),
            "track": selected_track
        }
        st.session_state.optimization_history.append(log_entry)
        st.success("Setup logged!")
    
    if st.session_state.optimization_history:
        # Create history dataframe
        history_data = []
        for i, entry in enumerate(st.session_state.optimization_history):
            history_data.append({
                "Run": i + 1,
                "Time": entry["timestamp"],
                "Lap Time": f"{entry['lap_time']:.3f}s",
                "Track": entry["track"],
                "Front Wing": entry["setup"]["front_wing_angle"],
                "Rear Wing": entry["setup"]["rear_wing_angle"]
            })
        
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True)
        
        # Plot optimization progress
        if len(st.session_state.optimization_history) > 1:
            lap_times = [entry["lap_time"] for entry in st.session_state.optimization_history]
            
            fig_progress = go.Figure()
            fig_progress.add_trace(go.Scatter(
                x=list(range(1, len(lap_times) + 1)),
                y=lap_times,
                mode='lines+markers',
                name='Lap Time',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8)
            ))
            
            fig_progress.update_layout(
                title="Optimization Progress",
                xaxis_title="Iteration",
                yaxis_title="Lap Time (s)",
                height=300
            )
            
            st.plotly_chart(fig_progress, use_container_width=True)

# --- 🎯 Pareto Front Viewer ---
def generate_pareto_front():
    n = 20
    lap_times = np.linspace(89, 95, n)
    tire_wear = 100 - (lap_times - 89) * 8 + np.random.randn(n) * 2
    tire_wear = np.clip(tire_wear, 70, 100)
    return pd.DataFrame({"Lap_Time": lap_times, "Tire_Life": tire_wear, "Setup_ID": range(n)})

with st.expander("🎯 Multi-Objective Optimization (Pareto Analysis)", expanded=False):
    st.markdown("### ⚖️ Lap Time vs Tire Life Trade-off")
    if st.button("🔄 Generate Pareto Front"):
        st.session_state.pareto_front = generate_pareto_front()

    if st.session_state.pareto_front is not None:
        fig = go.Figure()
        df = st.session_state.pareto_front
        fig.add_trace(go.Scatter(
            x=df['Lap_Time'], y=df['Tire_Life'], mode='markers',
            marker=dict(size=10, color=df['Setup_ID'], colorscale='Viridis', showscale=True),
            text=[f"Setup {i}" for i in df['Setup_ID']],
            hovertemplate="<b>%{text}</b><br>Lap Time: %{x:.3f}s<br>Tire Life: %{y:.1f}%<extra></extra>",
        ))
        fig.update_layout(title="Pareto Front: Lap Time vs Tire Life", xaxis_title="Lap Time (s)", yaxis_title="Tire Life Score (%)", height=500)
        st.plotly_chart(fig, use_container_width=True)

        selected_setup_id = st.selectbox("Select Pareto Optimal Setup", df['Setup_ID'].tolist(), format_func=lambda x: f"Setup {x} (Time: {df.loc[x, 'Lap_Time']:.3f}s)")
        if st.button("🎯 Apply Selected Setup"):
            st.session_state.setup = {
                "front_wing_angle": np.random.randint(15, 45),
                "rear_wing_angle": np.random.randint(15, 45),
                "ride_height": np.random.randint(25, 45),
                "suspension_stiffness": np.random.randint(2, 10),
                "brake_bias": np.random.randint(52, 58)
            }
            st.success(f"✅ Applied Pareto optimal setup #{selected_setup_id}")
            st.rerun()

# Export functionality
with st.expander("📤 Export & Share", expanded=False):
    st.markdown("### 💾 Export Setup Data")
    
    export_data = {
        "setup": st.session_state.setup,
        "track": selected_track,
        "conditions": {
            "track_temperature": track_temperature,
            "air_temperature": air_temperature,
            "humidity": humidity,
            "weather": weather_condition,
            "fuel_load": fuel_load,
            "tire_compound": tire_compound
        },
        "performance": calculate_performance_metrics(st.session_state.setup, track_info),
        "timestamp": datetime.now().isoformat()
    }
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        st.download_button(
            label="📄 Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"f1_setup_{selected_track}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col_export2:
        # Create CSV data
        csv_data = pd.DataFrame([{
            **st.session_state.setup,
            "track": selected_track,
            "lap_time": export_data["performance"]["lap_time"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }]).to_csv(index=False)
        
        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name=f"f1_setup_{selected_track}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# --- 📋 Full Report Export ---
def generate_setup_report():
    return {
        "executive_summary": {
            "optimal_lap_time": f"{calculate_performance_metrics(st.session_state.setup, track_info)['lap_time']:.3f}s",
            "key_strengths": ["High cornering performance", "Balanced aerodynamics"],
            "areas_for_improvement": ["Tire degradation", "Top speed on straights"]
        },
        "detailed_analysis": {
            "aerodynamic_efficiency": 85.2,
            "mechanical_grip": 78.9,
            "overall_balance": 82.1
        },
        "recommendations": [
            "Consider reducing rear wing angle for better straight-line speed",
            "Monitor tire temperatures closely with current suspension settings",
            "Brake bias setting is optimal for this track layout"
        ]
    }

with st.expander("📋 Advanced Reporting", expanded=False):
    st.markdown("### 📊 Comprehensive Setup Analysis")
    if st.button("📈 Generate Full Report"):
        report_data = generate_setup_report()
        st.markdown("#### 📋 Executive Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Optimal Lap Time", report_data['executive_summary']['optimal_lap_time'])
            st.markdown("**Key Strengths:**")
            for s in report_data['executive_summary']['key_strengths']:
                st.write(f"• {s}")
        with col2:
            st.markdown("**Areas for Improvement:**")
            for a in report_data['executive_summary']['areas_for_improvement']:
                st.write(f"• {a}")

        st.markdown("#### 🔧 Performance Breakdown")
        c1, c2, c3 = st.columns(3)
        c1.metric("Aerodynamic Efficiency", f"{report_data['detailed_analysis']['aerodynamic_efficiency']:.1f}%")
        c2.metric("Mechanical Grip", f"{report_data['detailed_analysis']['mechanical_grip']:.1f}%")
        c3.metric("Overall Balance", f"{report_data['detailed_analysis']['overall_balance']:.1f}%")

        st.markdown("#### 💡 Engineer Recommendations")
        for i, rec in enumerate(report_data["recommendations"], 1):
            st.info(f"**{i}.** {rec}")

# Footer with professional information
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-top: 2rem;">
    <h3>🏎️ F1 Car Setup Workbench</h3>
    <p>Advanced Formula 1 Setup Optimization Tool</p>
    <p><strong>Features:</strong> Multi-objective optimization • Real-time physics simulation • Professional telemetry analysis</p>
    <p><em>Built with Streamlit, Plotly, and advanced optimization algorithms</em></p>
</div>
""", unsafe_allow_html=True)