# 🏎️ F1 Car Setup Optimizer

A comprehensive Formula 1 car setup optimization platform that combines machine learning, physics simulation, and advanced optimization algorithms to find optimal car configurations for different tracks and racing conditions.

---

## 🎯 Project Focus
This project serves as a complete Formula 1 engineering workbench that addresses the complex challenge of car setup optimization through multiple AI-driven approaches.

### Core Problem
Formula 1 car setup involves balancing numerous conflicting parameters (aerodynamics, suspension, braking) to achieve optimal lap times while considering:
- **Track-specific characteristics** (Monaco vs Monza require completely different approaches)
- **Multi-objective tradeoffs** (lap time vs tire wear vs stability)
- **Dynamic conditions** (weather, fuel load, track temperature)
- **Strategic considerations** (qualifying vs race setup, grid position impact)

### Technical Approach
The platform implements three distinct optimization strategies:
- **Bayesian Optimization** – Efficient single-objective optimization using Gaussian processes
- **Multi-Objective Optimization (NSGA-II)** – Pareto frontier analysis for competing objectives
- **Reinforcement Learning** – PPO agent that learns optimal setups through environmental interaction

---

## ✨ Key Features

### 🛠️ Interactive Setup Workbench
- Real-time parameter adjustment: Front/Rear Wing angles, Ride Height, Suspension stiffness, Brake Bias
- Live performance visualization: Radar charts for Top Speed, Cornering, Stability, and Tire Life balance
- Physics-based telemetry simulation: Speed, brake, and throttle profiles over track distance
- Setup health monitoring: Anomaly detection for unsafe or suboptimal configurations

### 🤖 AI-Powered Optimization Engines
- Bayesian Optimization: Uses scikit-optimize for efficient parameter space exploration
- Pareto Front Analysis: Multi-objective optimization revealing lap time vs tire wear tradeoffs
- Reinforcement Learning Agent: Pre-trained PPO models adapt to track conditions and grid position
- Predictive Analytics: ML models for lap time prediction, anomaly detection, and maintenance risk assessment

### 🧠 Advanced Analysis & Explainability
- SHAP-based feature analysis: Understand which setup parameters impact performance most
- Setup comparison tools: Side-by-side analysis of different configurations
- Optimization history tracking: Evolution of setup improvements over time
- Professional reporting: Export detailed analysis reports in multiple formats

### 🌍 Comprehensive Track Database
- 20+ real F1 circuits with detailed characteristics
- Track-specific base times and difficulty ratings
- Optimal strategy recommendations for each circuit
- Environmental condition modeling (temperature, weather, humidity)
- Grid position impact analysis for strategic setup decisions

---

## 🛠️ Technical Architecture

| Component | Technology Stack |
|-----------|------------------|
| **Frontend** | Streamlit + Plotly for interactive visualizations |
| **Machine Learning** | scikit-learn, scikit-optimize, RandomForest, Isolation Forest |
| **Multi-Objective Optimization** | pymoo (NSGA-II algorithm) |
| **Reinforcement Learning** | Stable Baselines3 (PPO), Custom Gymnasium environment |
| **Physics Simulation** | Custom physics engine with aerodynamics and tire models |
| **Data Processing** | NumPy, Pandas for numerical computation and data handling |
| **Explainability** | SHAP for model interpretability |
| **Export & Reporting** | JSON, CSV, PDF report generation |

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+ (Required for Stable Baselines3 compatibility)
- pip package manager
- Git for cloning the repository

### Installation Steps

**1. Clone the repository**
```bash
git clone https://github.com/rembertdesigns/F1-car-setup-optimizer.git
cd F1-car-setup-optimizer
```
**2. Set up Python virtual environment**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```
**3. Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
**4. Create necessary directories**
```bash
mkdir -p data models/rl logs
```

## Initial Setup & Training

⚠️ **IMPORTANT**: The app requires trained models to function properly. Run these training scripts in order:

**1. Generate synthetic training data**
```bash
python src/simulate_physics_model.py
```
**2. Train core prediction models**
```bash
# Main lap time predictor (REQUIRED)
python src/train_model.py

# Anomaly detection model
python src/train_anomaly_model.py

# Maintenance risk predictor
python src/train_maintenance_model.py
```
**3. Train reinforcement learning agent (Optional but recommended)**
```bash
python src/train_rl_agent.py
```
Note: RL training takes 10-15 minutes depending on your hardware

**4. Launch the application**
```bash
streamlit run src/app.py
```

## Verification

The app should open in your browser at [http://localhost:8501](http://localhost:8501). You should see:

✅ No "Model not found" errors  
✅ All optimization modes working (Bayesian, NSGA-II, RL)  
✅ Performance predictions updating when you adjust parameters

---

## 📁 Project Structure
```
F1-car-setup-optimizer/
│
├── 📂 data/                              # Training data and logs
│   ├── synthetic_car_setup_v2.csv        # Generated training dataset
│   └── setup_log.csv                     # User-generated setup logs
│
├── 📂 logs/                              # Training logs and tensorboard data
│   └── rl/                               # RL training logs
│
├── 📂 models/                            # Trained ML models
│   ├── 📂 rl/                            # Reinforcement learning models
│   │   ├── ppo_car_setup_agent_*.zip     # Training checkpoints
│   │   └── ppo_car_setup_final.zip       # Final trained RL agent
│   ├── lap_time_predictor_v2.pkl         # Main prediction model
│   ├── setup_anomaly_detector.pkl        # Anomaly detection
│   ├── maintenance_risk_predictor.pkl    # Predictive maintenance
│   └── *.json                            # Feature configurations
│
├── 📂 src/                               # Source code
│   ├── 📂 envs/                          # RL environment
│   │   └── setup_env.py                  # Custom Gymnasium environment
│   ├── app.py                            # Main Streamlit application
│   ├── optimizer.py                      # Optimization algorithms
│   ├── physics_model.py                  # Physics simulation engine
│   ├── simulate_physics_model.py         # Data generation
│   ├── train_model.py                    # ML model training
│   ├── train_rl_agent.py                 # RL agent training
│   ├── train_anomaly_model.py            # Anomaly detection training
│   └── train_maintenance_model.py        # Maintenance model training
│
├── requirements.txt                      # Python dependencies
├── LICENSE                               # MIT License
└── README.md                             # This documentation
```
