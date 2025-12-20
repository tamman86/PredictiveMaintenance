import pandas as pd
import numpy as np
import yaml
import joblib
import matplotlib.pyplot as plt
import utils
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

### Loading yaml configuration ###
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

print(f"Target Dataset: {config['data']['train_file']}")
print(f"Whitelisted Sensors: {config['features']['whitelist']}")
window_size = config['features']['window_size']

### Load and Prep Data
# Define columns
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = config['features']['sensor_names']
col_names = index_names + setting_names + sensor_names

# Load Data
df = pd.read_csv(config['data']['train_file'], sep='\s+', header=None, names=col_names)

# Calculate Remaining Useful Life (RUL)
# RUL = (Max Cycle the engine reached) - (Current Cycle)
max_cycles = df.groupby('unit_nr')['time_cycles'].max().reset_index()
max_cycles.columns = ['unit_nr', 'max']

# Merge back into original dataframe
df = df.merge(max_cycles, on=['unit_nr'], how='left')

# Calculate RUL
df['RUL'] = df['max'] - df['time_cycles']

# Drop 'max' since its cheating with the answer
#df.drop('max', axis=1, inplace=True)

print(f"Data Loaded. Shape: {df.shape}")

### Feature Selection to remove irrelevant sensors ###

# Filter #1: Remove "Constant" sensors (Variance = 0)
def get_constant_sensors(df, sensors_list):
    constant_sensors = []
    for sensor in sensors_list:
        if df[sensor].std() <= 0.00001:
            constant_sensors.append(sensor)
    return constant_sensors

bad_sensors = get_constant_sensors(df, sensor_names)

# Filter #2: Remove "Low Correlation" sensors
correlation_threshold = config['features']['correlation_threshold']

correlations = df[sensor_names + ['RUL']].corr()['RUL']
low_corr_sensors = correlations[abs(correlations) < correlation_threshold].index.tolist()

# Apply whitelist to sensor removal list
whitelist = config['features']['whitelist']
bad_sensors = [s for s in bad_sensors if s not in whitelist]
low_corr_sensors = [s for s in low_corr_sensors if s not in whitelist]

# Finalize list of remaining sensors
sensors_to_use = [s for s in sensor_names if s not in bad_sensors and s not in low_corr_sensors]

print(f"\nFinal Selected Sensors: ({len(sensors_to_use)}):")
print(sensors_to_use)

############### Anomaly Detection Logic ###############
features = sensors_to_use
healthy_percentage = config['anomaly_detection']['healthy_percentage']

# Incorporating Dynamic Healthy Split
df['cutoff_cycle'] = df['max'] * healthy_percentage

train_data = df[df['time_cycles'] <= df['cutoff_cycle']]

X_train_healthy = train_data[features]
print(f"Training on {len(X_train_healthy)} rows.")
print(f"Defined as first {healthy_percentage*100}% of each engine's life.")

# Train Isolation Forest
iso_forest = IsolationForest(n_estimators = 100, contamination = config['anomaly_detection']['contamination'], random_state = 42)
iso_forest.fit(X_train_healthy)

'''
### Isolation Forest Test
unit_no = 4
unit_1 = df[df['unit_nr'] == unit_no].copy()
unit_1_features = unit_1[features]

anomaly_score = -1 * iso_forest.decision_function(unit_1_features)
unit_1['smoothed_score'] = pd.Series(anomaly_score, index=unit_1.index).rolling(window=10).mean()

plt.figure(figsize=(12, 6))
plt.plot(unit_1['time_cycles'], anomaly_score, label='Raw Score', color='purple', alpha=0.3)
plt.plot(unit_1['time_cycles'], unit_1['smoothed_score'], label='Smoothed Trend', color='purple', linewidth=2)
plt.axhline(0, color='red', linestyle='--', label='Threshold')
plt.title("Effect of Smoothing on False Positives")
plt.legend()
plt.show()
'''

############### Remaining Useful Life Logic ###############
### Implement Rolling Windows ###
df_processed, features_final = utils.engineer_features(df, sensors_to_use, window_size = window_size)
df_processed = df_processed.dropna()

print(f"New Data Shape: {df_processed.shape}")

### Training the model ###
X = df_processed[features_final]
y = df_processed['RUL']

# Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = config['training']['test_size'],
                                                    random_state = config['training']['random_state'])

# Train model
model = RandomForestRegressor(n_estimators = config['training']['n_estimators'],
                              random_state = config['training']['random_state'])
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Model RMSE: {rmse:.2f} cycles")   # On average, the prediciton is off by +/- {rmse} cycles


### Save Products ###
model_package = {
    'model': model,
    'features_final': features_final,
    'sensor_list': sensors_to_use,
    'window_size': window_size,
    'rmse': rmse,
    'config': config
}

output_file = config['output']['model_filename']
joblib.dump(model_package, output_file)
print(f"Model saved to {output_file}")

'''
### Volatility Threshold Tuning ###
# Setup healthy time vs failing time
healthy_data = df_processed[df_processed['RUL'] > config['parameters']['healthy_threshold']]
failing_data = df_processed[df_processed['RUL'] <= config['parameters']['warning_threshold']]

# Volatility analysis for selected sensor on healthy engines
mean_vol = healthy_data['s_7_std'].mean()
std_vol = healthy_data['s_7_std'].std()
max_vol_healthy = healthy_data['s_7_std'].max()

# Calculating thresholds (95% and 99.7%)
sigma_2 = mean_vol + (2 * std_vol)  # 95% Confidence for Warning
sigma_3 = mean_vol + (3 * std_vol)  # 99.7% Confidence for Critical


### Alerting Logic ###
# Take live sensor data, predict RUL, and issue alert if critical
def generate_alert(df_row_data, features_list, threshold_rul=config['parameters']['warning_threshold'],
                   threshold_volatility=sigma_2):
    input_data = df_row_data[features_list].to_frame().T # Reshape input to one row dataframe

    pred_rul = model.predict(input_data)[0]

    # Volatility Check
    volatility = input_data['s_7_std'].iloc[0]

    status = "GREEN"
    message = "System Normal"

    if pred_rul < config['parameters']['critical_threshold']:
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
print(generate_alert(data_stable_failing, features_final))

print("\n--- Case #2: High Volatility Event ---")
print(generate_alert(high_vol_row, features_final))
'''