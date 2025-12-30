### consumer.py

import sys
import asyncio
import json
import pandas as pd
import joblib
import asyncpg
from aiokafka import AIOKafkaConsumer

# Windows Fix
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Config
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "sensor_readings"
MODEL_FILE = "pdm_model_v2.pkl"

DB_CONFIG = {
    "user": "postgres",
    "password": "password",
    "database": "pdm_db",
    "host": "localhost",
    "port": "5432"
}


def flatten_readings(reading):
    flat_data = {
        'unit_nr': reading['unit_nr'],
        'time_cycles': reading['time_cycles']
    }
    for i, val in enumerate(reading['sensors']):
        flat_data[f's_{i + 1}'] = val
    for i, val in enumerate(reading['settings']):
        flat_data[f'setting_{i + 1}'] = val
    return flat_data


async def init_db(pool):
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS sensor_predictions (
        timestamp TIMESTAMPTZ DEFAULT NOW(),
        unit_nr INTEGER,
        time_cycles INTEGER,
        rul_prediction DOUBLE PRECISION,
        raw_data JSONB
    );
    '''
    create_hypertable_query = "SELECT create_hypertable('sensor_predictions', 'timestamp', if_not_exists => TRUE);"
    async with pool.acquire() as conn:
        await conn.execute(create_table_query)
        try:
            await conn.execute(create_hypertable_query)
        except Exception:
            pass


async def save_to_db(pool, flat_reading, prediction):
    query = """
    INSERT INTO sensor_predictions (unit_nr, time_cycles, rul_prediction, raw_data)
    VALUES ($1, $2, $3, $4)
    """
    raw_payload = flat_reading.copy()
    del raw_payload['unit_nr']
    del raw_payload['time_cycles']
    json_payload = json.dumps(raw_payload)

    async with pool.acquire() as conn:
        await conn.execute(query, flat_reading['unit_nr'], flat_reading['time_cycles'], float(prediction), json_payload)


async def process_message(reading, model_artifact, pool):
    # --- SAFETY CHECK: VALIDATE DATA ---
    if not isinstance(reading, dict):
        print(f"⚠️ SKIPPING BAD DATA (Not a Dict): {reading}")
        return

    if 'unit_nr' not in reading:
        print(f"⚠️ SKIPPING BAD DATA (Missing 'unit_nr'): {reading}")
        return
    # -----------------------------------

    unit_id = reading['unit_nr']
    flat_reading = flatten_readings(reading)

    # UNPACK ARTIFACT
    model = model_artifact['model']
    required_features = model_artifact['features']

    try:
        # Dynamic Feature Extraction
        input_data = {}
        missing_sensors = []

        for feature in required_features:
            val = flat_reading.get(feature)
            if val is not None:
                input_data[feature] = val
            else:
                missing_sensors.append(feature)

        if not missing_sensors:
            # Predict
            features_df = pd.DataFrame([input_data], columns=required_features)
            prediction = model.predict(features_df)[0]

            # Save
            await save_to_db(pool, flat_reading, prediction)

            if unit_id == 1 and flat_reading['time_cycles'] % 10 == 0:
                print(f"💾 Unit {unit_id} | Cycle {flat_reading['time_cycles']} | RUL: {prediction:.2f}")
        else:
            print(f"⚠️ Unit {unit_id} missing features: {missing_sensors}")

    except Exception as e:
        print(f"⚠️ Calculation Error on Unit {unit_id}: {e}")


async def consume():
    print(f"Loading model artifact from {MODEL_FILE}")
    try:
        model_artifact = joblib.load(MODEL_FILE)
        print(f"Model Loaded. Features: {model_artifact['features']}")
    except Exception as e:
        print(f"CRITICAL: Failed to load model. {e}")
        return

    print("Connecting to Database")
    try:
        pool = await asyncpg.create_pool(**DB_CONFIG)
        await init_db(pool)
    except Exception as e:
        print(f"Database Connection Failed: {e}")
        return

    consumer = AIOKafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset='latest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    await consumer.start()
    print("Consumer Listening...")

    try:
        async for msg in consumer:
            reading = msg.value
            await process_message(reading, model_artifact, pool)

    except Exception as e:
        print(f"Consumer Error: {e}")
    finally:
        await consumer.stop()
        await pool.close()


if __name__ == '__main__':
    asyncio.run(consume())