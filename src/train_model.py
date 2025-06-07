import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load updated dataset
df = pd.read_csv("data/synthetic_car_setup_v2.csv")

# Separate features and target
X = df.drop(columns=["lap_time"])
y = df["lap_time"]

# Save feature list for use in inference
model_features = X.columns.tolist()
with open("models/model_features.json", "w") as f:
    json.dump(model_features, f)
print("🧠 Feature list saved to models/model_features.json")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("✅ Model trained successfully!")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save trained model
joblib.dump(model, "models/lap_time_predictor_v2.pkl")
print("✅ Model saved to models/lap_time_predictor_v2.pkl")

# Plot feature importance
importances = model.feature_importances_
sorted_idx = np.argsort(importances)
plt.figure(figsize=(8, 5))
plt.barh(X.columns[sorted_idx], importances[sorted_idx], color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Feature Importances - Lap Time Predictor V2")
plt.tight_layout()
plt.savefig("models/feature_importance_v2.png")
print("📊 Feature importance plot saved to models/feature_importance_v2.png")