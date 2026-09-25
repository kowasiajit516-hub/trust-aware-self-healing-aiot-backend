import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
MODEL_OUTPUT_PATH = "random_forest.joblib"

def generate_synthetic_telemetry(num_samples=1000):
    """Generates synthetic sensor telemetry for model training when MongoDB is offline."""
    np.random.seed(42)
    
    # Features: [humidity (%), hour (0-23), minute (0-59), day_of_week (0-6), node_id_encoded (0-4)]
    humidity = np.random.uniform(30.0, 90.0, num_samples)
    hour = np.random.randint(0, 24, num_samples)
    minute = np.random.randint(0, 60, num_samples)
    day_of_week = np.random.randint(0, 7, num_samples)
    node_id = np.random.randint(0, 5, num_samples)
    
    # Target: temperature (°C) correlated with humidity and hour
    temperature = 25.0 - (humidity * 0.1) + (hour * 0.3) + np.random.normal(0, 0.5, num_samples)
    
    X = pd.DataFrame({
        'humidity': humidity,
        'hour': hour,
        'minute': minute,
        'day_of_week': day_of_week,
        'node_id_encoded': node_id
    })
    y = pd.Series(temperature, name='temperature')
    
    return X, y

def train_and_save_model():
    print("Preparing training dataset...")
    X, y = generate_synthetic_telemetry()

    print(f"Training RandomForestRegressor on {len(X)} samples...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)

    # Evaluate
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    
    print(f"✓ Model Training Complete.")
    print(f"  - R² Score: {r2:.4f}")
    print(f"  - RMSE:     {rmse:.4f}")

    # Serialize Model Artifact
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"✓ Successfully saved artifact to: {os.path.abspath(MODEL_OUTPUT_PATH)}")

if __name__ == "__main__":
    train_and_save_model()