import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

### Load and Prep Data
# Define columns
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1,22)]
col_names = index_names + setting_names + sensor_names

# Load Data
df = pd.read_csv('CMaps/test_FD001.txt', sep='\s+', header=None, names=col_names)

# Calculate Remaining Useful Life (RUL)
# RUL = (Max Cycle the engine reached) - (Current Cycle)
max_cycles = df.groupby('unit_nr')['time_cycles'].max().reset_index()
max_cycles.columns = ['unit_nr', 'max']

# Merge back into original dataframe
df = df.merge(max_cycles, on=['unit_nr'], how='left')

# Calculate RUL
df['RUL'] = df['max'] - df['time_cycles']

# Drop 'max' since its cheating with the answer
df.drop('max', axis=1, inplace=True)

print(f"Data Loaded. Shape: {df.shape}")
print(df[['unit_nr', 'time_cycles', 's_2', 's_14', 'RUL']].head())

### Feature Selection to remove irrelevant sensors ###

# Create a whitelist to protect certain sensors regardless of Feature Selection
do_not_remove_list = ['s_2', 's_14']

# Filter #1: Remove "Constant" sensors (Variance = 0)
def get_constant_sensors(df, sensors_list):
    constant_sensors = []
    for sensor in sensors_list:
        if df[sensor].std() <= 0.00001:
            constant_sensors.append(sensor)
    return constant_sensors

all_sensors = ['s_{}'.format(i) for i in range(1,22)]
bad_sensors = get_constant_sensors(df, all_sensors)
print(f"Dropping Constant Sensors: {bad_sensors}")

# Filter #2: Remove "Low Correlation" sensors
correlation_threshold = 0.05

correlations = df[all_sensors + ['RUL']].corr()['RUL']
low_corr_sensors = correlations[abs(correlations) < correlation_threshold].index.tolist()
print(f"Dropping Low Correlation Sensors: {low_corr_sensors}")

# Apply whitelist to sensor removal list
bad_sensors = [s for s in bad_sensors if s not in do_not_remove_list]
low_corr_sensors = [s for s in all_sensors if s not in do_not_remove_list]

# Finalize list of remaining sensors
sensors_to_use = [s for s in all_sensors if s not in bad_sensors and s not in low_corr_sensors]

print(f"\nFinal Selected Sensors: ({len(sensors_to_use)}):")
print(sensors_to_use)


### Implement Rolling Windows ###
# Define window size
window_size = 10

# Create rolling mean
df_rolling = df.groupby('unit_nr')[sensors_to_use].rolling(window=window_size).mean()
df_rolling.columns = [f"{col}_mean" for col in sensors_to_use]
df_rolling = df_rolling.reset_index(level=0, drop=True)

# Create rolling standard deviation
df_std = df.groupby('unit_nr')[sensors_to_use].rolling(window=window_size).std()
df_std.columns = [f"{col}_std" for col in sensors_to_use]
df_std = df_std.reset_index(level=0, drop=True)

df_processed = pd.concat([df, df_rolling, df_std], axis = 1)

df_processed = df_processed.dropna()

print(f"New Data Shape: {df_processed.shape}")
print("Added columns like: s_2_mean and s_2_std")

### Training the model ###
features = sensors_to_use + list(df_rolling.columns) + list(df_std.columns)
X = df_processed[features]
y = df_processed['RUL']

# Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Model RMSE: {rmse:.2f} cycles")   # On average, the prediciton is off by +/- {rmse} cycles

### Volatility Threshold Tuning ###
# Setup healthy time vs failing time
healthy_data = df_processed[df_processed['RUL'] > 100]
failing_data = df_processed[df_processed['RUL'] <= 30]

# Volatility analysis for selected sensor on healthy engines
mean_vol = healthy_data['s_7_std'].mean()
std_vol = healthy_data['s_7_std'].std()
max_vol_healthy = healthy_data['s_7_std'].max()

# Calculating thresholds (95% and 99.7%)
sigma_2 = mean_vol + (2 * std_vol)  # 95% Confidence for Warning
sigma_3 = mean_vol + (3 * std_vol)  # 99.7% Confidence for Critical


### Alerting Logic ###
# Take live sensor data, predict RUL, and issue alert if critical
def generate_alert(df_row_data, features_list, threshold_rul=30, threshold_volatility=sigma_2):
    input_data = df_row_data[features_list].to_frame().T # Reshape input to one row dataframe

    pred_rul = model.predict(input_data)[0]

    # Volatility Check
    volatility = input_data['s_7_std'].iloc[0]

    status = "GREEN"
    message = "System Normal"

    if pred_rul < 15:
        status = "RED"
        message = f"CRITICAL FAILURE IMMINENT. Predicted RUL: {round(pred_rul, 1)} cycles."
    elif pred_rul < threshold_rul:
        status = "YELLOW"
        message = f"Warning: Approaching End of Life. Predicted RUL: {round(pred_rul, 1)} cycles."

    if volatility > threshold_volatility:
        if status != "RED" and volatility < sigma_3:
            status = "YELLOW-VOLATILITY"
            message = f"⚠️ Volatility Spike Detected! High instability in sensor s_7 (Std: {volatility:.4f}). Pred RUL: {round(pred_rul, 1)}."
        elif status != "RED" and volatility >= sigma_3:
            status = "RED-VOLATILITY"
            message = f"⚠️ Volatility Spike Detected! Critical instability in sensor s_7 (Std: {volatility:.4f}). Pred RUL: {round(pred_rul, 1)}."

    return {
        "Predicted_RUL": round(pred_rul, 1),
        "Volatility_s7": round(volatility, 3),
        "Status": status,
        "Message": message
    }

### Simulating alert ###
# 1. Simulator: Grab data from an engine that is stable but near its end (low RUL)
# Unit 7 is known to fail around cycle 160
data_stable_failing = df_processed[(df_processed['unit_nr']==7) & (df_processed['time_cycles']==158)].iloc[0]

# 2. Early Volatility
high_vol_row = df_processed.loc[df_processed['s_7_std'].idxmax()]

print("\n--- Case #1: Stable but Failing ---")
print(generate_alert(data_stable_failing, features))

print("\n--- Case #2: High Volatility Event ---")
print(generate_alert(high_vol_row, features))