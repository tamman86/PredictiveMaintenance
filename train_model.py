import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold

# Config
DATA_FILE = "training_data.csv"
MODEL_FILE = "pdm_model_generic.pkl"
RUL_CLIP_LIMIT = None

def add_rul_label(df, id_col='unit_nr', time_col='time_cycles', clip_value=None):
    # Group by Unit to find max cycle
    max_cycles = df.groupby(id_col)[time_col].max()

    # Merge "max_life" into main dataframe
    df = df.merge(max_cycles.to_frame(name='max_life'), left_on=id_col, right_index=True)

    # Calculate RUL (Max - Current)
    df['RUL'] = df['max_life'] - df[time_col]

    # Apply Clipping
    if clip_value is not None:
        print(f"Clipping RUL at {clip_value} cycles")
        df['RUL'] = df['RUL'].clip(upper=clip_value)

    # Clean up helper column
    df = df.drop(columns=['max_life'])

    return df

def auto_train():
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded dataset with {len(df.columns)} columns")
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_FILE}")
        return

    # 2. Inject RUL column if needed
    if 'RUL' not in df.columns:
        df = add_rul_label(df, id_col='unit_nr', time_col = 'time_cycles', clip_value=RUL_CLIP_LIMIT)
    else:
        print("'RUL' column already exists. Skipping generation")

    # 3. Identify Potential Features
    exclude_cols = ['unit_nr', 'RUL']

    potential_features = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

    X = df[potential_features]
    y = df["RUL"]

    # 4. Remove "Flat" Sensors
    selector_var = VarianceThreshold(threshold = 0)
    selector_var.fit(X)

    clean_features = X.columns[selector_var.get_support()].tolist()
    dropped_flat = len(potential_features) - len(clean_features)
    if dropped_flat > 0:
        print(f"Dropped {dropped_flat} flat sensors")

    # 5. Correlation Check
    # Calculate correlation with RUL. Sensors with Correlation < 0.05 will be dropped
    correlations = df[clean_features].corrwith(df['RUL']).abs()
    final_features = correlations[correlations > 0.05].index.tolist()

    dropped_noise = len(clean_features) - len(final_features)
    if dropped_noise > 0:
        print(f"Dropped {dropped_noise} noisy (Low Correlation) sensors")

    print(f"✅ Final Feature Set ({len(final_features)}): {final_features}")

    # 6. Train Model
    print(f"Training Random Forest on: {final_features}")
    X_train, X_test, y_train, y_test = train_test_split(df[final_features], y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth = 15, random_state=42)
    model.fit(X_train, y_train)

    # 7. Evaluate
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Model Performance (RMSE): {rmse:.2f} cycles")

    # 8. Save Artifact
    model_artifact = {
        "model": model,
        "features": final_features,
        "version": "3.0_auto_clipped"
    }
    joblib.dump(model_artifact, MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")

if __name__ == "__main__":
    auto_train()
