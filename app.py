### app.py

import streamlit as st
import pandas as pd
import psycopg2
import time
from datetime import datetime, timedelta, timezone
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION ---
REFRESH_INTERVAL = 1.0  # Time between refreshing display
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

def load_data_since(unit_id, start_dt):
    formatted_time = start_dt.strftime("%Y-%m-%d %H:%M:%Sz")

    query = f"""
        SELECT timestamp, time_cycles, rul_prediction, raw_data 
        FROM sensor_predictions 
        WHERE unit_nr = {unit_id}
          AND timestamp >= '{formatted_time}'
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

def get_data_range_date(unit_id, start_dt, end_dt):
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

# Filter Mode Selector
filter_mode = st.sidebar.radio(
    "View Mode:",
    ["Live Monitor (Minutes)", "Deep Dive (Hours)", "Historical Analysis"]
)

fragment_args = {}

if filter_mode == "Live Monitor (Minutes)":
    # Slider for Minutes
    minutes = st.sidebar.slider("Show Last X Minutes", min_value=1, max_value=60, value=5)
    fragment_args = {"mode": "live", "delta": timedelta(minutes=minutes)}

elif filter_mode == "Deep Dive (Hours)":
    # Slider for Hours
    hours = st.sidebar.slider("Show Last X Hours", min_value=1, max_value=24, value=1)
    fragment_args = {"mode": "live", "delta": timedelta(hours=hours)}

elif filter_mode == "Historical Analysis":
    # Date Picker for older data
    d = st.sidebar.date_input("Select Date", datetime.now())
    fragment_args = {"mode": "history", "date": d}

# 2. Main Display
@st.fragment(run_every=REFRESH_INTERVAL)
def show_live_dashboard(unit_id, args):
    if not args:
        return

    now_utc = datetime.now(timezone.utc)
    df = pd.DataFrame()

    if args["mode"] == "live":
        # LIVE MODE: Start = Now - Delta. No End Limit.
        start_dt = now_utc - args["delta"]
        df = load_data_since(unit_id, start_dt)

    elif args["mode"] == "history":
        # HISTORICAL MODE: Specific 24h window
        sel_date = args["date"]
        start_dt = datetime.combine(sel_date, datetime.min.time())
        end_dt = datetime.combine(sel_date, datetime.max.time())
        df = get_data_range_date(unit_id, start_dt, end_dt)

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
                    delta_color="normal" if latest_rul > 72 else "inverse")

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
            st.plotly_chart(fig_rul, use_container_width=True)

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

            st.plotly_chart(fig_sensors, use_container_width=True)
    else:
        st.info("No data found in this window. Check the generator or select a different time.")

# 3. Main Execution
if selected_unit and fragment_args:
    show_live_dashboard(selected_unit, fragment_args)