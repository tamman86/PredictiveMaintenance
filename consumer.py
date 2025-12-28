import sys
import asyncio
import json
import pandas as pd
import joblib
from aiokafka import AIOKafkaConsumer
import utils

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Config
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "sensor_readings"
MODEL_FILE = "pdm_model_v1.pkl"

# State Memory (To remember last "n" cycles for each sensor"
# Format: {unit_id: [reading_1, reading_2, ... reading_n] }
sensor_buffers = {}

def flatten_readings(reading):
    flat_data = {
        'unit_nr': reading['unit_nr'],
        'time_cycles': reading['time_cycles']
    }

    # Unpacking Sensors
    for i, val in enumerate(reading['sensors']):
        flat_data[f's_{i+1}'] = val

    for i, val in enumerate(reading['settings']):
        flat_data[f'setting_{i+1}'] = val

    return flat_data

async def process_message(reading, model_package):
    unit_id = reading['unit_nr']

    model = model_package['model']
    required_sensors = model_package['sensor_list']
    window_size = model_package['window_size']

    flat_reading = flatten_readings(reading)

    # 1. Initialize buffer if new unit
    if unit_id not in sensor_buffers:
        sensor_buffers[unit_id] = []

    # 2. Adding new reading to memory
    sensor_buffers[unit_id].append(flat_reading)

    # 3. Maintain window sized buffer
    if len(sensor_buffers[unit_id]) > window_size:
        sensor_buffers[unit_id].pop(0)

    # 4. Verify we have a full window to make prediction
    if len(sensor_buffers[unit_id]) == window_size:
        df_window = pd.DataFrame(sensor_buffers[unit_id])

        try:
            # Engineer Features
            df_engineered, final_cols = utils.engineer_features(df_window, required_sensors, window_size)

            latest_row = df_engineered.iloc[[-1]][final_cols]

            # Prediction
            prediction = model.predict(latest_row)[0]

            print(f"🔮 Unit {unit_id} | Cycle {reading['time_cycles']} | RUL Prediction: {prediction:.2f}")
        except Exception as e:
            print(f"⚠️ Math Error on Unit {unit_id}: {e}")

    else:
        # Not enough data yet (Need "window_size", have X)
        print(f"⏳ Unit {unit_id} | Buffering... ({len(sensor_buffers[unit_id])}/{window_size})")

async def consume():
    # Load Model
    print(f"Loading model from {MODEL_FILE}")
    try:
        model_package = joblib.load(MODEL_FILE)
        print("Model Loaded")
    except FileNotFoundError:
        print("Model not found!")
        return

    # Connect to Kafka
    consumer = AIOKafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset='latest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    await consumer.start()
    print("Consumer Listening...")

    try:
        # Main Loop
        async for msg in consumer:
            reading = msg.value
            await process_message(reading, model_package)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await consumer.stop()

if __name__ =='__main__':
    asyncio.run(consume())
