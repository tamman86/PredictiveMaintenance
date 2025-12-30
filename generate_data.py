### generate_data.py
'''
import time
import random
import requests
import threading
import json

API_URL = "http://localhost:8000/ingest"

# Config Simulation
engines = {}
for i in range(1,11):
    engines[i] = {
        "cycle": 0,
        "wear": 0.0,
        "wear_rate": random.uniform(0.5,2.0) if i < 4 else random.uniform(0.1, 0.5)
    }

def generate_sensor_reading(unit_id):
    engine = engines[unit_id]
    engine["cycle"] += 1
    engine["wear"] += engine["wear_rate"]

    # Simulate "Failure"
    if engine["wear"] > 250:
        print(f"Engine {unit_id} failed! Replaced with new engine")
        engine["cycle"] = 0
        engine["wear"] = 0.0

    # Simulate "wear" sensors [Temp(s_11) up, Pressure(s_12) down]
    s_11_base = 520.0
    s_12_base = 642.0

    # Apply degradation
    s_11 = s_11_base + (engine["wear"] * 5.0) + random.normalvariate(0, 2)
    s_12 = s_12_base - (engine["wear"] * 4.0) + random.normalvariate(0, 2)

    # 19 other sensors of random noise
    other_sensors = [random.uniform(500,700) for _ in range(19)]

    # Construct full list
    sensors = other_sensors[:10] + [s_11, s_12] + other_sensors[10:]

    payload = {
        "unit_nr": unit_id,
        "time_cycles": engine["cycle"],
        "settings": [random.uniform(0.,1) for _ in range(3)],
        "sensors": sensors
    }
    return payload

print("Starting physics simulator...")
counter = 1
while True:
    for unit_id in range(1, 11):
        data = generate_sensor_reading(unit_id)
        print(counter)
        counter += 1
        try:
            requests.post(API_URL, json=data)
            # Occasionally print status
            if unit_id == 1 and data['time_cycles'] % 10 == 0:
                print(f"Unit 1 Cycle: {data['time_cycles']} | Wear: {engines[1]['wear']:.2f}")
        except Exception as e:
            print(f"API Error: {e}")

    time.sleep(0.1)
'''

import time
import random
import requests
import threading  # We are finally using this!
import json

API_URL = "http://localhost:8000/ingest"

# Config Simulation
# We move the engine state initialization inside the thread function usually,
# or keep it global. Global is fine for this size.
engines = {}
for i in range(1, 11):
    engines[i] = {
        "cycle": 0,
        "wear": 0.0,
        "wear_rate": random.uniform(0.5, 2.0) if i < 4 else random.uniform(0.1, 0.5)
    }


def simulate_engine(unit_id):
    """
    This function runs FOREVER in its own thread.
    It simulates exactly one engine.
    """
    # Create a Session for this thread (Reuse TCP connection = MUCH Faster)
    session = requests.Session()

    print(f"🚀 Unit {unit_id} Started")

    while True:
        engine = engines[unit_id]
        engine["cycle"] += 1
        engine["wear"] += engine["wear_rate"]

        # Simulate "Failure"
        if engine["wear"] > 250:
            print(f"💥 Engine {unit_id} FAILED! Resetting...")
            engine["cycle"] = 0
            engine["wear"] = 0.0

        # Simulate "wear" sensors
        s_11_base = 520.0
        s_12_base = 642.0

        s_11 = s_11_base + (engine["wear"] * 5.0) + random.normalvariate(0, 2)
        s_12 = s_12_base - (engine["wear"] * 4.0) + random.normalvariate(0, 2)

        # Other sensors
        other_sensors = [random.uniform(500, 700) for _ in range(19)]
        sensors = other_sensors[:10] + [s_11, s_12] + other_sensors[10:]

        payload = {
            "unit_nr": unit_id,
            "time_cycles": engine["cycle"],
            "settings": [random.uniform(0., 1) for _ in range(3)],
            "sensors": sensors
        }

        try:
            # Send data using the persistent session
            session.post(API_URL, json=payload)

            # Print logic (Reduced spam)
            if unit_id == 1 and engine['cycle'] % 10 == 0:
                print(f"Unit 1 Cycle: {engine['cycle']} | Wear: {engine['wear']:.2f}")

        except Exception as e:
            print(f"⚠️ Unit {unit_id} Connection Error: {e}")

        # Throttle THIS engine only
        # 0.5s is a good speed for visual demos.
        # Since they run in parallel, you get 20 events/sec total (10 units * 2 events/sec)
        time.sleep(0.5)

    # --- MAIN EXECUTION ---


print("Starting Parallel Physics Simulator...")

threads = []
for i in range(1, 11):
    # distinct thread for each unit
    t = threading.Thread(target=simulate_engine, args=(i,))
    t.daemon = True  # Kills threads if you close the main script
    t.start()
    threads.append(t)

# Keep the main script alive so the threads don't die
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping Simulator...")