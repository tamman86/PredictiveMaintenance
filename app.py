import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from sklearn.ensemble import IsolationForest

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Critical RUL number
crit_RUL = config['parameters']['critical_RUL']

### Loading Data ###
@st.cache_data
def load_data():
    # 1. Define Columns
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # 2. Load Data
    df = pd.read_csv(config['data']['train_file'], sep='\s+', header=None, names=col_names)


    # 3. Calculate RUL
    max_cycles = df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_nr', 'max']
    df = df.merge(max_cycles, on = ['unit_nr'], how = 'left')
    df['RUL'] = df['max'] - df['time_cycles']


    # 4. Anomaly Scores
    subset_features = ['s_2', 's_3', 's_4', 's_6', 's_7', 's_8', 's_9', 's_11',
                       's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']

    healthy_data = df[df['time_cycles'] <= 50][subset_features]

    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(healthy_data)
    df['anomaly_score'] = -1 * iso.decision_function(df[subset_features])


    model = load_model()
    if hasattr(model, 'predict'):
        sensors = subset_features

        # Fast Rolling Window
        grouped = df.groupby('unit_nr')[sensors]
        roll_mean = grouped.rolling(window = 10).mean().reset_index(0, drop = True)
        roll_std = grouped.rolling(window = 10).std().reset_index(0, drop = True)

        roll_mean.columns = [f"{c}_mean" for c in sensors]
        roll_std.columns = [f"{c}_std" for c in sensors]

        df_feats = pd.concat([df, roll_mean, roll_std], axis=1)

        features_list = sensors + list(roll_mean.columns) + list(roll_std.columns)

        valid_rows = df_feats.dropna(subset=features_list).index

        df['predicted_RUL'] = np.nan
        df.loc[valid_rows, 'predicted_RUL'] = model.predict(df_feats.loc[valid_rows, features_list])

        # Backfill first predictions before rolling window is established
        df['predicted_RUL'] = df.groupby('unit_nr')['predicted_RUL'].bfill().fillna(0)

    return df

@st.cache_resource
def load_model():
    try:
        loaded_package = joblib.load(config['output']['model_filename'])
        return loaded_package['model']
    except FileNotFoundError:
        st.error("Error: 'pdm_model_v1.pkl' not found. Did you run train_model.py?")
        return None


@st.cache_data
def calculate_fleet_performance(df, _model):
    # 1. Re-Engineer Features for the whole dataset
    sensors = ['s_2', 's_3', 's_4', 's_6', 's_7', 's_8', 's_9', 's_11',
               's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']

    # Calculate rolling features
    df_poly = df.copy()
    window = 10

    # Fast vectorized rolling calculation
    grouped = df_poly.groupby('unit_nr')[sensors]
    roll_mean = grouped.rolling(window=window).mean().reset_index(0, drop=True)
    roll_std = grouped.rolling(window=window).std().reset_index(0, drop=True)

    # Rename columns
    roll_mean.columns = [f"{c}_mean" for c in sensors]
    roll_std.columns = [f"{c}_std" for c in sensors]

    # Join back
    df_poly = pd.concat([df_poly, roll_mean, roll_std], axis=1).dropna()

    # 2. Prepare Feature List
    features_final = sensors + list(roll_mean.columns) + list(roll_std.columns)

    # 3. Batch Predict
    if hasattr(_model, 'predict'):
        df_poly['predicted_RUL'] = _model.predict(df_poly[features_final])

        # 4. Calculate Error (RMSE) per Unit
        # Error = (Predicted - Actual)^2
        df_poly['error_squared'] = (df_poly['predicted_RUL'] - df_poly['RUL']) ** 2

        # Group by Unit to get MSE, then sqrt for RMSE
        performance_df = df_poly.groupby('unit_nr')['error_squared'].mean().pow(0.5).reset_index()
        performance_df.columns = ['unit_nr', 'RMSE']

        # 5. Categorize
        def grade_model(rmse):
            if rmse < 20: return "A (Excellent)"
            if rmse < 35: return "B (Good)"
            if rmse < 50: return "C (Fair)"
            return "D (Poor)"

        performance_df['Grade'] = performance_df['RMSE'].apply(grade_model)

        return performance_df
    else:
        return pd.DataFrame()  # Empty if no model

df = load_data()
model = load_model()

### Sidebar Design ###
st.sidebar.title("Plant Monitor")
current_cycle = st.sidebar.slider("Simulate Time (Cycle):", min_value = 1, max_value = 360, value = 100)

unit_ids = df['unit_nr'].unique()
selected_unit = st.sidebar.selectbox("Select Unit:", ["Fleet Overview"] + list(unit_ids))

### Dashboard Logic ###
if selected_unit == "Fleet Overview":
    st.title(f"Fleet Status at Cycle {current_cycle}")

    # Logic:
    # 1. If engine max_life < current_cycle -> It's Dead. Show 0.
    # 2. If engine is alive -> Show the 'predicted_RUL' from that specific cycle.

    # Create grid for latest status of each engine
    status_values = np.zeros(100)

    # Get the max lifespan of every unit for the check
    unit_lifespans = df.groupby('unit_nr')['max'].max()

    for unit_id in unit_ids:
        max_life = unit_lifespans[unit_id]

        if current_cycle > max_life:
            # Case A: The engine has exploded.
            # Even if the last prediction was 5 cycles, we flatline it to 0.
            current_val = 0
        else:
            # Case B: The engine is alive.
            # Get the prediction for this specific cycle
            try:
                row = df[(df['unit_nr'] == unit_id) & (df['time_cycles'] == current_cycle)]
                if not row.empty:
                    current_val = row['predicted_RUL'].values[0]
                else:
                    current_val = 0  # Should not happen if logic is correct
            except:
                current_val = 0

        status_values[int(unit_id) - 1] = current_val

    grid_status = status_values.reshape(10, 10)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Green = High RUL and Red = Low RUL
    sns.heatmap(grid_status,
                cmap="RdYlGn",
                annot=True,
                fmt=".0f",
                ax=ax,
                vmin = 0,
                vmax = 150,
                cbar_kws={'label': 'Remaining Useful Life'})
    st.pyplot(fig)

    st.info("👆 Click the Sidebar to drill down into a specific Red Unit.")

    ### Accuracy report ###
    st.divider()

    # MODEL HEALTH REPORT
    st.subheader("🤖 Model Performance Report")

    # Calculate performance (Cached)
    perf_df = calculate_fleet_performance(df, model)

    if not perf_df.empty:
        # 1. The Summary Cards
        grade_counts = perf_df['Grade'].value_counts()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Excellent (RMSE < 20)", grade_counts.get("A (Excellent)", 0))
        k2.metric("Good (RMSE < 35)", grade_counts.get("B (Good)", 0))
        k3.metric("Fair (RMSE < 50)", grade_counts.get("C (Fair)", 0))
        k4.metric("Poor (RMSE > 50)", grade_counts.get("D (Poor)", 0))

        # 2. The Interactive Scatter Plot
        # Allows user to spot the "Bad" engines visually
        st.markdown("#### Accuracy by Unit")

        # Color code points based on Grade
        fig_perf, ax_perf = plt.subplots(figsize=(10, 4))

        # We scatter plot Unit ID vs RMSE
        sns.scatterplot(data=perf_df, x='unit_nr', y='RMSE', hue='Grade',
                        palette={'A (Excellent)': 'green', 'B (Good)': 'blue', 'C (Fair)': 'orange', 'D (Poor)': 'red'},
                        ax=ax_perf)

        ax_perf.axhline(35, color='grey', linestyle='--', label='Acceptable Threshold (35)')
        ax_perf.set_title("Which Engines is the AI struggling with?")
        ax_perf.set_xlabel("Unit ID")
        ax_perf.set_ylabel("Error (Cycles)")
        st.pyplot(fig_perf)

        # 3. The "Worst Offenders" List
        # This answers your question: "Which ones spiked?"
        with st.expander("Show Details for 'Poor' Performing Units"):
            bad_units = perf_df[perf_df['Grade'] == "D (Poor)"]
            if not bad_units.empty:
                st.write(bad_units.sort_values('RMSE', ascending=False))
                st.caption("Tip: Select these Unit IDs in the sidebar to investigate why the predictions spiked.")
            else:
                st.success("No units found with Poor performance!")

# Unit Detail View for selecting fleet item
else:
    st.title(f"Diagnostics: Unit #{selected_unit}")

    # Data Filtration
    unit_data = df[df['unit_nr'] == selected_unit].copy()

    # Smoothing Calculation
    # Check for 'anomaly_score' column
    if 'anomaly_score' in unit_data.columns:
        # Calculate smoothed score
        unit_data['smoothed_score'] = unit_data['anomaly_score'].rolling(window=10, min_periods=1).mean()
    else:
        unit_data['smoothed_score'] = np.nan

    latest_data = unit_data.iloc[-1]

    # Top Status Bar
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Cycle", int(latest_data['time_cycles']))
    c2.metric("Predicted RUL", f"{int(latest_data['RUL'])} cycles")

    # Dynamic Status Logic
    status = "🟢 Normal"
    if latest_data['smoothed_score'] > 0: status = "🔴 Anomaly Detected"
    elif latest_data['RUL'] < 30: status = "🟡 Warning"
    c3.metric("System Status", status)

    # Chart 1: Anomaly Score
    st.subheader("Anomaly Detection Stream (Raw vs Smoothed)")

    fig1, ax1 = plt.subplots(figsize=(10, 4))

    # Plot Raw Score
    ax1.plot(unit_data['time_cycles'],
             unit_data['anomaly_score'],
             color = 'purple',
             alpha = 0.3,
             label = 'Raw Score')

    # Plot Smoothed Trend
    ax1.plot(unit_data['time_cycles'],
             unit_data['smoothed_score'],
             color = 'purple',
             linewidth = 2,
             label = 'Smoothed Trend (Window = 10)')

    ax1.axhline(0, color='red', linestyle='--', label='Anomaly Threshold')
    ax1.set_title(f"Engine #{selected_unit}: Anomaly Score vs. Time Cycles")
    ax1.set_xlabel("Time Cycles")
    ax1.set_ylabel("Anomaly Score (Higher is Worse)")
    ax1.legend()
    st.pyplot(fig1)

    # CHART 2: Sensor Telemetry
    st.subheader("Live Sensor Telemetry")
    sensor_to_plot = st.selectbox("Select Sensor:", ['s_2', 's_2_mean', 's_7_std'])
    st.line_chart(unit_data[['time_cycles', sensor_to_plot]].set_index('time_cycles'))

    # CHART 3: RUL vs True RUL (The Truth Check)
    st.subheader("Model Performance: Predicted vs. Actual")

    # 1. PREPARE DATA FOR PREDICTION
    # We need to recreate the rolling features for the whole history of this unit
    # so the model can make a prediction for every single cycle.

    # Define the sensors your model needs (Must match your training config!)
    sensors_for_features = ['s_2', 's_3', 's_4', 's_6', 's_7', 's_8', 's_9', 's_11',
                            's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']

    # Create a temporary copy to calculate features on the fly
    history_df = unit_data.copy()

    # Calculate Rolling Means and Stds (Window=10)
    # This matches exactly what we did in 'train_model.py'
    window = 10
    for col in sensors_for_features:
        history_df[f"{col}_mean"] = history_df[col].rolling(window=window).mean()
        history_df[f"{col}_std"] = history_df[col].rolling(window=window).std()

    # Drop the 'Warm up' rows (NaNs) where we can't make predictions
    history_df = history_df.dropna()

    # 2. GENERATE PREDICTIONS
    # Check if we have the list of expected features from the pickle file
    # If using dummy model, we skip this.
    if hasattr(model, 'predict'):
        try:
            # We must select ONLY the columns the model was trained on, in the EXACT order.
            # Usually, we get this list from loaded_package['features']
            # Here, we assume 'features_final' is available or we construct it:
            features_final = sensors_for_features + \
                             [f"{c}_mean" for c in sensors_for_features] + \
                             [f"{c}_std" for c in sensors_for_features]

            # Predict for the entire history
            history_df['predicted_RUL'] = model.predict(history_df[features_final])

            # 3. PLOT
            fig2, ax2 = plt.subplots(figsize=(10, 4))

            # Ground Truth (Black Dashed)
            ax2.plot(unit_data['time_cycles'], unit_data['RUL'],
                     color='black', linestyle='--', label='True RUL (Actual)')

            # Prediction (Orange Line)
            ax2.plot(history_df['time_cycles'], history_df['predicted_RUL'],
                     color='orange', linewidth=2, label='Model Prediction')

            ax2.set_title(f"Engine #{selected_unit}: Lifecycle Prediction Accuracy")
            ax2.set_xlabel("Time Cycles")
            ax2.set_ylabel("Remaining Useful Life (Cycles)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"Could not generate predictions. Error: {e}")
            st.warning("Make sure your dashboard is loading the Real Model, not the Dummy Model.")
    else:
        st.warning("Model not loaded or is a dummy. Cannot plot predictions.")