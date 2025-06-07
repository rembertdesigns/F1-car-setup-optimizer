# 🏎️ F1 Car Setup Optimizer

An interactive web application that uses Bayesian Optimization to find the optimal Formula 1 car setup for specific tracks and performance goals. This tool acts as a "Setup Workbench," allowing users to balance trade-offs between raw lap time, tire preservation, and handling.

---

## ✨ Key Features

* **Interactive Setup Workbench:** A modern UI built with Streamlit where users can adjust car setup parameters (wing angles, ride height, suspension, brake bias) using interactive sliders.
* **Live Performance Profile:** A dynamic radar chart provides immediate visual feedback on the car's balance as sliders are adjusted, showing the trade-offs between Top Speed, Cornering Grip, Stability, and Tire Life.
* **Context-Aware Optimization:**
    * **Track Selection:** Choose from classic F1 tracks like Monza, Monaco, and Spa, each with unique characteristics and a visual track map.
    * **Condition Control:** Adjust parameters like track temperature and grip level to see how they influence the optimal setup.
* **Bayesian Optimization Engine:**
    * Utilizes `scikit-optimize` to intelligently search the vast parameter space for the best setup.
    * Instead of brute-force, it efficiently hones in on high-performing setups based on what it learns.
* **Multi-Objective Tradeoffs:**
    * Users can define what "optimal" means to them by adjusting weights for three key goals: `Lap Time`, `Tire Preservation`, and `Handling Balance`.
    * The optimizer finds the best setup that satisfies the user's specified priorities.

---

## 🛠️ Technologies Used

* **Core:** Python 3.10+
* **Optimization & Machine Learning:** `scikit-optimize` (for Bayesian Optimization), `scikit-learn` & `joblib` (for the underlying lap time prediction model).
* **Data Handling:** NumPy, Pandas
* **User Interface & Visualization:** Streamlit, Plotly

---

## 🧱 Project Structure

```bash
F1-car-setup-optimizer/
│
├── models/
│   └── lap_time_predictor.pkl   # Pre-trained ML model that predicts lap time from setup
│
├── src/
│   ├── app.py                   # The main Streamlit application script
│   └── optimizer.py             # Contains the Bayesian optimization logic
│
├── requirements.txt             # Python package dependencies
└── README.md
```
