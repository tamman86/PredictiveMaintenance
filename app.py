### app.py

import streamlit as st
import pandas as pd
import psycopg2
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION ---
REFRESH_INTERVAL = 2.0  # Time between refreshing display
DB_CONFIG = {
    "dbname": "pdm_db",
    "user": "postgres",
    "password": "password",
    "host": "localhost",
    "port": "5432"
}


# --- DATABASE FUNCTIONS ---
def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def load_data(unit_id, limit=1000):
    """Fetch the history for a specific engine ordered by TIME"""
    query = f"""
    SELECT timestamp, time_cycles, rul_prediction, raw_data 
    FROM sensor_predictions 
    WHERE unit_nr = {unit_id}
    ORDER BY timestamp ASC
    """

    # Fetch all and slice in Pandas to ensure we get the true "tail" of time
    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        local_tz = datetime.now().astimezone().tzinfo
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        df['timestamp'] = df['timestamp'].dt.tz_convert(local_tz)

    return df

def get_active_units():
    conn = get_connection()
    # Query to get units that have reported recently
    query = "SELECT DISTINCT unit_nr FROM sensor_predictions ORDER BY unit_nr"
    df = pd.read_sql(query, conn)
    conn.close()
    return df['unit_nr'].tolist()

# --- DASHBOARD LAYOUT ---
st.set_page_config(page_title="Real-Time PDM", layout="wide")
st.title("✈️ Predictive Maintenance Live Twin")

# 1. Sidebar
st.sidebar.header("Controls")
available_units = get_active_units()

if not available_units:
    st.warning("Waiting for data... (Start the Generator!)")
    time.sleep(2)
    st.rerun()

selected_unit = st.sidebar.selectbox("Select Engine Unit", available_units)
history_depth = st.sidebar.slider("History Depth (Records)", min_value=50, max_value=2000, value=300)

# 2. Main Display
@st.fragment(run_every=REFRESH_INTERVAL)
def show_live_dashboard(unit_id, depth):

    # Load Data
    df = load_data(unit_id)

    if not df.empty:
        # Slice to requested depth
        df_view = df.tail(depth)

        # --- PARSE RAW SENSORS ---
        # Use s_11 and s_12 to visualize the physics
        df_view = df_view.copy()
        df_view['Temp (s_11)'] = df_view['raw_data'].apply(lambda x: x.get('s_11', 0))
        df_view['Pressure (s_12)'] = df_view['raw_data'].apply(lambda x: x.get('s_12', 0))

        latest_row = df_view.iloc[-1]
        latest_rul = latest_row['rul_prediction']
        latest_cycle = latest_row['time_cycles']

        latest_time = latest_row['timestamp'].strftime('%H:%M:%S')

        # 3-Stage Health Status
        if latest_rul > 100:
            status_text = "🟢 Healthy"
            color = "normal"
        elif latest_rul > 50:
            status_text = "🟡 Warning"
            color = "off"
        else:
            status_text = "🔴 CRITICAL FAILURE"
            color = "inverse"

        # --- KPI ROW ---
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Latest Update", f"{latest_time}")

        # Color logic: Green if > 50, Red if < 50
        kpi2.metric("Est. Operating Hours Left", f"{latest_rul:.0f} Hours",
                    delta_color="normal" if latest_rul > 50 else "inverse")

        kpi3.metric("Status", status_text)

        # --- CHARTS ROW ---
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Predictive Maintenance Window")
            fig_rul = px.line(df_view, x="timestamp", y="rul_prediction",
                              title="Remaining Useful Life (Hours)",
                              labels={"rul_prediction": "Hours until Failure"})

            # Add Threshold Line
            fig_rul.add_hline(y=0, line_dash="solid", line_color="red")
            fig_rul.update_yaxes(range=[-10, 300])  # Range of RUL chart (Min/Max)
            st.plotly_chart(fig_rul, width="content")

        with c2:
            st.subheader("Sensor Correlation (Temp vs Pressure)")

            # Dual-axis chart to show the physics correlation
            fig_sensors = go.Figure()

            # Temp on Left Y-Axis
            fig_sensors.add_trace(go.Scatter(
                x=df_view['timestamp'],
                y=df_view['Temp (s_11)'],
                name="Temp (s_11)",
                line=dict(color="#EF553B")
            ))

            # Pressure on Right Y-Axis
            fig_sensors.add_trace(go.Scatter(
                x=df_view['timestamp'],
                y=df_view['Pressure (s_12)'],
                name="Pressure (s_12)",
                line=dict(color="#636EFA"),
                yaxis="y2"
            ))

            fig_sensors.update_layout(
                yaxis=dict(title="Temperature"),
                yaxis2=dict(title="Pressure", overlaying="y", side="right"),
                title="Physics Fingerprint",
                legend=dict(x=0, y=1.1, orientation="h")
            )

            st.plotly_chart(fig_sensors, width="content")

# 3. Main Execution
if selected_unit:
    show_live_dashboard(selected_unit, history_depth)