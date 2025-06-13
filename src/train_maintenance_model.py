# src/train_maintenance_model.py
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load synthetic dataset
df = pd.read_csv("data/synthetic_car_setup_v2.csv")

# Simulate wear risk labels based on harsh setups
def simulate_wear(row):
    risk = 0
    risk += (row["suspension_stiffness"] / 11) * 0.4
    risk += abs(row["brake_bias"] - 55) / 10 * 0.2
    risk += abs(row["ride_height"] - 30) / 20 * 0.2
    risk += 1 - row.get("grip_level", 1.0) * 0.2
    return 1 if risk > 0.5 else 0

df["wear_risk"] = df.apply(simulate_wear, axis=1)

FEATURES = [
    "front_wing_angle", "rear_wing_angle", "ride_height",
    "suspension_stiffness", "brake_bias",
    "track_temperature", "grip_level"
]

X = df[FEATURES]
y = df["wear_risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "models/maintenance_risk_predictor.pkl")

with open("models/maintenance_features.json", "w") as f:
    json.dump(FEATURES, f)

print("✅ Maintenance model saved.")
