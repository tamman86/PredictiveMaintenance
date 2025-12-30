### app.py

import streamlit as st
import pandas as pd
import psycopg2
import time
from datetime import datetime, timedelta
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

def load_data_by_date(unit_id, start_dt, end_dt):
    query = f"""
        SELECT timestamp, time_cycles, rul_prediction, raw_data 
        FROM sensor_predictions 
        WHERE unit_nr = {unit_id}
          AND timestamp >= '{start_dt}'
          AND timestamp <= '{end_dt}'
        ORDER BY timestamp ASC
        """
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

def get_data_range():
    conn = get_connection()
    query = "SELECT MIN(timestamp), MAX(timestamp) FROM sensor_predictions"
    df = pd.read_sql(query, conn)
    conn.close()
    if not df.empty and df.iloc[0,0] is not None:
        return df.iloc[0,0], df.iloc[0,1]
    return datetime.now(), datetime.now()

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

st.sidebar.markdown("---")
st.sidebar.subheader("Time Filter")

# Filter Mode Selector
filter_mode = st.sidebar.radio(
    "Select Range Mode:",
    ["Quick Slider (Days)", "Custom Lookback", "Specific Date Range"]
)

now = datetime.now()
start_dt = now
end_dt = now

if filter_mode == "Quick Slider (Days)":
    # Slider for 0 to 14 Days
    days_back = st.sidebar.slider("Show Last X Days", min_value=1, max_value=14, value=1)
    start_dt = now - timedelta(days=days_back)
    end_dt = now

elif filter_mode == "Custom Lookback":
    # Custom Integer Input
    days_back = st.sidebar.number_input("Enter Days to Look Back", min_value=1, value=30, step=1)
    start_dt = now - timedelta(days=days_back)
    end_dt = now

elif filter_mode == "Specific Date Range":
    # Specify Graph Window
    # Check the Database for what information is available
    min_db, max_db = get_data_range()

    default_start = max_db - timedelta(days=7) if max_db else now - timedelta(days=7)

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value = (default_start, max_db),
        max_value = datetime.now()
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
    else:
        st.sidebar.warning("Please select a Start and End date")

# 2. Main Display
@st.fragment(run_every=REFRESH_INTERVAL)
def show_live_dashboard(unit_id, start, end):

    # Load Data
    df = load_data_by_date(unit_id, start, end)

    if not df.empty:
        # Parse Sensors
        df['Temp (s_11)'] = df['raw_data'].apply(lambda x: x.get('s_11', 0))
        df['Pressure (s_12)'] = df['raw_data'].apply(lambda x: x.get('s_12', 0))

        latest_row = df.iloc[-1]
        latest_rul = latest_row['rul_prediction']
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
            fig_rul = px.line(df, x="timestamp", y="rul_prediction",
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
                x=df['timestamp'],
                y=df['Temp (s_11)'],
                name="Temp (s_11)",
                line=dict(color="#EF553B")
            ))

            # Pressure on Right Y-Axis
            fig_sensors.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['Pressure (s_12)'],
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
    else:
        st.info(f"No data found for Unit {unit_id} in the selected time range.")

# 3. Main Execution
if selected_unit:
    show_live_dashboard(selected_unit, start_dt, end_dt)