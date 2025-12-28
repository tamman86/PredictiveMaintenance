import time
import random
import requests
import threading

API_URL = "http://localhost:8000/ingest"
NUM_UNITS = 10          # Run 10 engines at once
SPEED_DELAY = 0.01      # Wait time between sends

def simulate_engine(unit_id):
    cycle = 1
    while True:
        # Generate fake data
        payload = {
            "unit_nr": unit_id,
            "time_cycles": cycle,
            "settings": [random.random() for _ in range(3)],
            "sensors": [random.uniform(500, 700) for _ in range(21)]
        }

        # Send to API
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
        except Exception as e:
            print(f"Connection Failed: {e}")
            break

        cycle += 1
        time.sleep(SPEED_DELAY)

# Start up threads (One per engine)
print(f"Starting simulation with {NUM_UNITS} engines...")
threads = []
for i in range(1, NUM_UNITS + 1):
    t = threading.Thread(target=simulate_engine, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
