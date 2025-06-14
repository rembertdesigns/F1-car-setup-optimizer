# 🏎️ F1 Car Setup Optimizer

An interactive web application that acts as an F1 engineering "workbench," using Bayesian Optimization, multi-objective tradeoffs, and reinforcement learning to find optimal car setups. Built with a full ML pipeline and sleek Streamlit UI.

---

## ✨ Key Features

### 🛠️ Setup Workbench
- Adjust parameters: Front/Rear Wing, Ride Height, Suspension, Brake Bias
- Radar chart visualizes balance: Top Speed, Cornering, Stability, Tire Life
- Simulated telemetry chart (Speed, Brake, Throttle over distance)

### 🤖 AI Optimization Modes
- **Bayesian Optimization** with `scikit-optimize`
- **Pareto Front Discovery** (multi-objective with `NSGA-II`)
- **Reinforcement Learning Agent** (trained PPO models using `stable-baselines3`)
- Setup suggestions adapt to track and starting grid position

### 🧠 Explainability & Diagnostics
- SHAP-based analysis of feature impact on lap time
- Anomaly detection for unsafe or extreme setup combinations
- Predictive maintenance model estimates mechanical risk

### 🔬 Simulation + Analysis Tools
- Save and compare setups in Slot A vs Slot B
- Track-specific base lap time, strategy tips, and performance summary
- Export options: JSON, CSV, text summary
- Optimization history with evolution charts

---

## 🛠️ Tech Stack

| Component                | Tech Used                                |
|-------------------------|-------------------------------------------|
| UI                      | Streamlit + Plotly                        |
| ML & Optimization       | scikit-learn, scikit-optimize, pymoo, shap |
| RL Agent                | Stable Baselines3 (PPO)                   |
| Data Processing         | NumPy, Pandas                             |
| Report Generation       | `fpdf2`, JSON, CSV                        |

---

## 🧱 Folder Structure

```bash
F1-car-setup-optimizer/
│
├── data/
│   ├── synthetic_car_setup_v2.csv
│   └── setup_log.csv             # Optional: user-generated logs
│
├── logs/                         # Training logs
│
├── models/
│   ├── rl/
│   │   ├── ppo_car_setup_agent_10000_steps.zip
│   │   ├── ...
│   │   └── ppo_car_setup_final.zip
│   ├── lap_time_predictor_v2.pkl
│   ├── setup_anomaly_detector.pkl
│   ├── maintenance_risk_predictor.pkl
│   ├── *.json (feature configs for each model)
│
├── src/
│   ├── app.py                   # Streamlit frontend
│   ├── optimizer.py             # Optimizers: Bayesian, Pareto, NSGA-II
│   ├── physics_model.py         # Physics-based lap simulation
│   ├── setup_env.py             # Gym-compatible environment for RL
│   ├── simulate_physics_model.py
│   ├── preprocess_data.py       # Generates synthetic training data
│   ├── train_model.py           # Lap time regressor training
│   ├── train_rl_agent.py        # PPO RL agent training
│   ├── train_anomaly_model.py
│   └── train_maintenance_model.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10 or newer.
* `pip` for package installation.

### Installation
```bash
git clone https://github.com/rembertdesigns/F1-car-setup-optimizer.git
cd F1-car-setup-optimizer
    
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```
### Run the App
```bash
streamlit run src/app.py
```
If models are missing, run:
```bash
python src/train_model.py               # For lap time predictor
python src/train_anomaly_model.py       # For anomaly detection
python src/train_maintenance_model.py   # For predictive maintenance
python src/train_rl_agent.py            # For PPO setup agent
```

---

## 🛣️ Future Enhancements

* **Full Race Weekend Simulation** (Practice, Quali, Race + tire degradation)
* Telemetry Import from Real F1 Data (FastF1 support)
* Manual vs AI Setup Battle Mode
* Setup Explainability with SHAP
* Physics-based lap sim w/ downforce + drag models

---

## 📄 License

This project is licensed under the MIT License.
