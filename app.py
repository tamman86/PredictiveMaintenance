import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import utils
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

@st.cache_resource
def load_model_package():
    try:
        return joblib.load(config['output']['model_filename'])
    except FileNotFoundError:
        return None

# Critical RUL number
crit_RUL = config['parameters']['critical_RUL']

### Loading Data ###
@st.cache_data
def load_data():
    # 1. Define Columns
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = config['features']['sensor_names']
    col_names = index_names + setting_names + sensor_names

    # 2. Load Data
    df = pd.read_csv(config['data']['train_file'], sep='\s+', header=None, names=col_names)


    # 3. Calculate RUL
    max_cycles = df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_nr', 'max']
    df = df.merge(max_cycles, on = ['unit_nr'], how = 'left')
    df['RUL'] = df['max'] - df['time_cycles']

    package = load_model_package()

    if package:
        model = package['model']
        saved_sensors = package['sensor_list']
        window_size = package.get('window_size', 10)

        # 4. Anomaly Detection
        healthy_data = df[df['time_cycles'] <= 50][saved_sensors]

        iso = IsolationForest(contamination=0.01, random_state=42)
        iso.fit(healthy_data)
        df['anomaly_score'] = -1 * iso.decision_function(df[saved_sensors])

        # RUL Prediction
        if hasattr(model, 'predict'):
            df_engineered, final_cols = utils.engineer_features(df, saved_sensors, window_size)

            valid_rows = df_engineered.dropna(subset = final_cols).index

            df['predicted_RUL'] = np.nan
            df.loc[valid_rows, 'predicted_RUL'] = model.predict(df_engineered.loc[valid_rows, final_cols])

            df['predicted_RUL'] = df.groupby('unit_nr')['predicted_RUL'].bfill().fillna(0)

    return df

@st.cache_data
def calculate_fleet_performance(df, _package):
    if not _package: return pd.DataFrame()

    model = _package['model']
    saved_sensors = _package['sensor_list']
    window_size = _package['window_size']

    # Prepare data for model
    df_poly, features_final = utils.engineer_features(df, saved_sensors, window_size)
    df_poly = df_poly.dropna(subset = features_final)

    if hasattr(model, 'predict'):
        df_poly['predicted_RUL'] = model.predict(df_poly[features_final])

        # Calculate RMSE
        df_poly['error_squared'] = (df_poly['predicted_RUL'] - df_poly['RUL']) ** 2

        # RMSE per Unit
        perf_df = df_poly.groupby('unit_nr')['error_squared'].mean().pow(0.5).reset_index()
        perf_df.columns = ['unit_nr', 'RMSE']

        def grade_model(rmse):
            if rmse < 20: return "A (Excellent)"
            if rmse < 35: return "B (Good)"
            if rmse < 50: return "C (Fair)"
            return "D (Poor)"

        perf_df['Grade'] = perf_df['RMSE'].apply(grade_model)
        return perf_df

    return pd.DataFrame()

# Main app logic
package = load_model_package()
if not package:
    st.error("Model file not found. Please run DataInput first")
    st.stop()

df = load_data()


### Sidebar Design ###
st.sidebar.title("Plant Monitor")

if 'sim_cycle' not in st.session_state: st.session_state.sim_cycle = 100

def increment_cycle():
    if st.session_state.sim_cycle < 360: st.session_state.sim_cycle += 1
def decrement_cycle():
    if st.session_state.sim_cycle > 1: st.session_state.sim_cycle -= 1

st.sidebar.markdown("### Time Travel Controls")
c_prev, c_slider, c_next = st.sidebar.columns([1, 4, 1])
with c_prev: st.button("◀", on_click=decrement_cycle)
with c_next: st.button("▶", on_click=increment_cycle)
with c_slider:
    current_cycle = st.slider("Cycle", 1, 360, key='sim_cycle', label_visibility="collapsed")
st.sidebar.caption(f"Current Cycle: **{current_cycle}**")

unit_ids = df['unit_nr'].unique()
selected_unit = st.sidebar.selectbox("Select Unit:", ["Fleet Overview"] + list(unit_ids))

### Dashboard Logic ###
if selected_unit == "Fleet Overview":
    st.title(f"Fleet Status Replay: Cycle {current_cycle}")

    # Logic:
    # 1. If engine max_life < current_cycle -> It's Dead. Show 0.
    # 2. If engine is alive -> Show the 'predicted_RUL' from that specific cycle.

    # Heatmap Logic
    status_values = np.zeros(100)
    unit_lifespans = df.groupby('unit_nr')['max'].max()

    for unit_id in unit_ids:
        # Check if dead
        if current_cycle > unit_lifespans[unit_id]:
            current_val = 0
        else:
            try:
                # Grab the pre-calculated prediction
                val = df[(df['unit_nr'] == unit_id) & (df['time_cycles'] == current_cycle)]['predicted_RUL'].values[0]
                current_val = val
            except:
                current_val = 0
        status_values[int(unit_id) - 1] = current_val

    grid = status_values.reshape(10, 10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(grid, cmap="RdYlGn", annot=True, fmt=".0f", vmin=0, vmax=150, ax=ax,
                cbar_kws={'label': 'Predicted RUL'})
    st.pyplot(fig)

    st.info("👆 Click the Sidebar to drill down into a specific Red Unit.")

    # MODEL PERFORMANCE REPORT
    st.divider()
    st.subheader("🤖 Model Performance Report")

    perf_df = calculate_fleet_performance(df, package)

    if not perf_df.empty:
        # Metric Cards
        grade_counts = perf_df['Grade'].value_counts()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Excellent (<20)", grade_counts.get("A (Excellent)", 0))
        k2.metric("Good (<35)", grade_counts.get("B (Good)", 0))
        k3.metric("Fair (<50)", grade_counts.get("C (Fair)", 0))
        k4.metric("Poor (>50)", grade_counts.get("D (Poor)", 0))

        # Scatter Plot
        st.markdown("#### Accuracy by Unit")
        fig_perf, ax_perf = plt.subplots(figsize=(10, 4))
        sns.scatterplot(data=perf_df, x='unit_nr', y='RMSE', hue='Grade',
                        palette={'A (Excellent)': 'green', 'B (Good)': 'blue', 'C (Fair)': 'orange', 'D (Poor)': 'red'},
                        ax=ax_perf)
        ax_perf.axhline(35, color='grey', linestyle='--', label='Threshold')
        st.pyplot(fig_perf)

        # Bad Apple List
        with st.expander("Show Details for 'Poor' Performing Units"):
            bad_units = perf_df[perf_df['Grade'] == "D (Poor)"]
            if not bad_units.empty:
                st.write(bad_units.sort_values('RMSE', ascending=False))

# UNIT DETAIL VIEW
else:
    st.title(f"Diagnostics: Unit #{selected_unit}")

    # Filter data for this unit
    unit_data = df[df['unit_nr'] == selected_unit].copy()

    # 1. Calculate Smoothed Score (Visualization only)
    if 'anomaly_score' in unit_data.columns:
        unit_data['smoothed_score'] = unit_data['anomaly_score'].rolling(window=10, min_periods=1).mean()
    else:
        unit_data['smoothed_score'] = np.nan

    # Check logic: Is the unit alive at this cycle?
    if current_cycle > unit_data['time_cycles'].max():
        st.error(f"Unit #{selected_unit} has failed (Max Cycle: {unit_data['time_cycles'].max()}).")
    else:
        # Slice data to "Current Time"
        current_view = unit_data[unit_data['time_cycles'] <= current_cycle]
        latest_data = current_view.iloc[-1]

        # Top Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Cycle", int(latest_data['time_cycles']))
        c2.metric("Predicted RUL", f"{int(latest_data['predicted_RUL'])} cycles")

        status = "🟢 Normal"
        if latest_data['smoothed_score'] > 0:
            status = "🔴 Anomaly Detected"
        elif latest_data['predicted_RUL'] < 30:
            status = "🟡 Warning"
        c3.metric("System Status", status)

        # Chart 1: Anomaly Score
        st.subheader("Anomaly Detection Stream")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(current_view['time_cycles'], current_view['anomaly_score'], color='purple', alpha=0.3, label='Raw')
        ax1.plot(current_view['time_cycles'], current_view['smoothed_score'], color='purple', linewidth=2,
                 label='Smoothed')
        ax1.axhline(0, color='red', linestyle='--', label='Threshold')
        ax1.legend()
        st.pyplot(fig1)

        # Chart 2: Telemetry
        st.subheader("Live Sensor Telemetry")
        # Dynamic selectbox based on available sensors
        available_sensors = package['sensor_list']
        sensor_to_plot = st.selectbox("Select Sensor:", available_sensors)
        st.line_chart(current_view[['time_cycles', sensor_to_plot]].set_index('time_cycles'))

        # Chart 3: Prediction vs Actual
        st.subheader("Model Validation (Predicted vs Actual)")

        # We need to generate predictions for the WHOLE history to draw the line
        # Use Utils for single-unit processing
        history_df, features_final = utils.engineer_features(unit_data, package['sensor_list'], package['window_size'])
        history_df = history_df.dropna(subset=features_final)

        # Predict
        if hasattr(package['model'], 'predict'):
            history_df['predicted_RUL'] = package['model'].predict(history_df[features_final])

            # Slice to current time for the plot
            history_view = history_df[history_df['time_cycles'] <= current_cycle]

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(unit_data['time_cycles'], unit_data['RUL'], color='black', linestyle='--', label='Ground Truth')
            ax2.plot(history_view['time_cycles'], history_view['predicted_RUL'], color='orange', linewidth=2,
                     label='Prediction')
            ax2.legend()
            ax2.set_ylabel("RUL")
            ax2.set_xlabel("Time Cycles")
            st.pyplot(fig2)