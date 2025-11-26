import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

### Visualize Sensor 2 for Unit 1 ###
# Trying to visualize high Temp -> decreased RUL
engine_no = 11   # Enter engine you'd like to chart (1 - 100)
plt.figure(figsize = (12,6))
unit_1 = df[df['unit_nr'] == engine_no]
plt.plot(unit_1['time_cycles'], unit_1['s_2'], label='Sensor ' + str(engine_no) + ' (Temp)')
plt.title('Sensor ' + str(engine_no) + ' Reading over Lifetime (Unit 1)')
plt.xlabel('Time (Cycles)')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Correlation Matrix: Which sensors predict RUL the best?
correlations = df.corr()['RUL'].sort_values()
print("Top 5 Sensors correlated with Failure:")
print(correlations.head(5))

### Setup modeling and alerting ###
# Remove all columns that don't give sensor info
drop_cols = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3', 'RUL']
features = [c for c in df.columns if c not in drop_cols]

X = df[features]
Y = df['RUL']

# Training and Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state =42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Model RMSE: {rmse:.2f} cycles")   # On average, the prediciton is off by +/- {rmse} cycles

# Visualize the distribution of Engine Lifespans
plt.figure(figsize=(10,4))
sns.histplot(max_cycles['max'], kde=True, bins=20)
plt.title('Distribution of Engine Lifespans (FD001)')
plt.xlabel('Cycles until Failure')
plt.show()

### Alerting Logic ###
# Take live sensor data, predict RUL, and issue alert if critical
def generate_alert(current_sensor_readings, threshold=30):
    input_data = pd.DataFrame([current_sensor_readings], columns=features)   # Reshape input
    pred_rul = model.predict(input_data)[0]

    status = "GREEN"
    message = "System Normal"

    if pred_rul < 15:
        status = "RED"
        message = "CRITICAL FAILURE IMMINENT - SCHEDULE MAINTENANCE IMMEDIATELY"
    elif pred_rul < threshold:
        status = "YELLOW"
        message = "Warning: Approaching End of Life"

    return {
        "Predicted_RUL": round(pred_rul, 1),
        "Status": status,
        "Message": message
    }

failing_engine_data = df[(df['unit_nr']==1) & (df['time_cycles']==31)][features].iloc[0]
print(failing_engine_data)
alert = generate_alert(failing_engine_data)

print("\n--- ALERT SYSTEM OUTPUT ---")
print(alert)