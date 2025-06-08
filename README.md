# 🏎️ F1 Car Setup Optimizer

An interactive F1 engineering simulation that lets you **optimize**, **analyze**, and **compare** car setups with AI-backed recommendations. Ideal for aspiring motorsport engineers, data scientists, or anyone looking to build smart, simulation-based apps.

## 🚀 Live Demo
**Coming Soon** – Streamlit Community Cloud deployment in progress.

---

## 📌 Overview

This app simulates F1 car setup tradeoffs using a custom ML model trained on synthetic data. It supports:

- Fast lap time predictions
- Tradeoff optimization (speed vs tire preservation vs handling)
- Interactive telemetry simulation
- Setup comparisons (Slot A vs Slot B)
- Full-track picker with real F1 circuits
- Sensitivity Analysis of setup variables

---

## 🧠 Key Features

### 🔧 Setup Workbench
Adjust front/rear wings, ride height, suspension, and brake bias to fine-tune your car.

### 🧪 AI-Powered Optimization
Bayesian Optimization finds the best setup for your chosen strategy:
- Focus on fastest lap
- Prioritize tire conservation
- Balance cornering vs straight-line speed

### 📈 Setup Tradeoff Visualizer
Run **Pareto Optimization** to view the tradeoff curve between lap time and tire life.

### 📊 Sensitivity Analysis
Explore how changing a single parameter affects lap time — is your setup stable or "knife-edge"?

### 📍 Track-Specific Insights
Choose from 20+ real F1 tracks with custom base lap times and descriptions.

### 📉 Setup Comparison Mode
Save and compare two setups side-by-side with radar charts and telemetry overlays.

---

## 🛠️ Technologies Used

- **Python + Streamlit**: Interactive UI & app framework
- **Scikit-learn**: ML model (Random Forest Regressor)
- **Scikit-optimize (skopt)**: Bayesian Optimization
- **Plotly**: Interactive visualizations
- **NumPy, Pandas**: Data manipulation
- **Joblib**: Model serialization

---

## 🧰 Setup & Installation

```bash
git clone https://github.com/rembertdesigns/F1-car-setup-optimizer.git
cd F1-car-setup-optimizer

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python3 src/simulate_physics_model.py

# Train model
python3 src/train_model.py

# Run the app
streamlit run app.py
```

---

## 📂 Project Structure

```bash
├── app.py                     # Main Streamlit app
├── src/
│   ├── train_model.py         # ML model training script
│   ├── simulate_physics_model.py # Data generation
│   └── optimizer.py           # Optimization logic
├── models/                    # Trained ML model and plots
├── data/                      # Synthetic dataset
├── README.md
```
