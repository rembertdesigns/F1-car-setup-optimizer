import pandas as pd
import joblib
import json
from sklearn.ensemble import IsolationForest

# Load synthetic dataset
df = pd.read_csv("data/synthetic_car_setup_v2.csv")

# Use features relevant to setup + race context
FEATURES = [
    'front_wing_angle', 'rear_wing_angle', 'ride_height',
    'suspension_stiffness', 'brake_bias',
    'track_temperature', 'grip_level',
    'fuel_weight', 'traffic'
]

# Train Isolation Forest
X = df[FEATURES]
model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
model.fit(X)

# Save model + feature list
joblib.dump(model, "models/setup_anomaly_detector.pkl")
with open("models/anomaly_features.json", "w") as f:
    json.dump(FEATURES, f)

print("✅ Anomaly detection model trained and saved.")

