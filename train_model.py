import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# 1. Load Data
df = pd.read_csv("training_data.csv")

# 2. Configure Features (INCLUDING TIME_CYCLES)
features = ['s_11', 's_12', 'time_cycles']
target = 'RUL'

X = df[features]
y = df[target]

# 3. Train
print(f"Training on {len(df)} records...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
print(f"Model RMSE: {rmse:.2f} cycles")

# 5. SAVE AS DICTIONARY
model_artifact = {
    "model": model,
    "features": features,
    "version": "2.1"
}

joblib.dump(model_artifact, 'pdm_model_v2.pkl')
print(f"SUCCESS: Saved model artifact. Type: {type(model_artifact)}")