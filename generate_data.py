### generate_data.py

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

    '''
    ### DEBUGGER ###
    if unit_id == 1:
        print(f"DEBUG: Unit 1 Wear Total: {engine['wear']:.2f}")
    if unit_id == 3:
        print(f"DEBUG: Unit 3 Wear Total: {engine['wear']:.2f}")
    if unit_id == 7:
        print(f"DEBUG: Unit 7 Wear Total: {engine['wear']:.2f}")
    if unit_id == 9:
        print(f"DEBUG: Unit 9 Wear Total: {engine['wear']:.2f}")
    '''

    payload = {
        "unit_nr": unit_id,
        "time_cycles": engine["cycle"],
        "settings": [random.uniform(0.,1) for _ in range(3)],
        "sensors": sensors
    }
    return payload

print("Starting physics simulator...")

while True:
    for unit_id in range(1, 11):
        data = generate_sensor_reading(unit_id)
        print("1")
        try:
            requests.post(API_URL, json=data)
            # Occasionally print status
            if unit_id == 1 and data['time_cycles'] % 10 == 0:
                print(f"Unit 1 Cycle: {data['time_cycles']} | Wear: {engines[1]['wear']:.2f}")
        except Exception as e:
            print(f"API Error: {e}")

    time.sleep(0.1)