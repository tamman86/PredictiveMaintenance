import pandas as pd
import random
import numpy as np


def simulate_engine_life(unit_id):
    # Units 1-3 are "Fast" (Lemon), others are "Slow" (Tank)
    # Randomize for the training set to get good coverage
    if random.random() < 0.3:
        wear_rate = random.uniform(0.5, 2.0)
    else:
        wear_rate = random.uniform(0.1, 0.5)

    wear = 0.0
    cycle = 0
    data = []

    # Run until failure (wear > 250)
    while wear <= 250:
        cycle += 1
        wear += wear_rate

        # Base operating values
        s_11_base = 520.0
        s_12_base = 642.0

        s_11 = s_11_base + (wear * 5.0) + random.normalvariate(0, 2)
        s_12 = s_12_base - (wear * 4.0) + random.normalvariate(0, 2)

        # Calculate Ground Truth RUL
        # RUL = (Max Wear - Current Wear) / Wear Rate
        remaining_wear = 250 - wear
        rul = int(remaining_wear / wear_rate)
        if rul < 0: rul = 0

        data.append({
            "unit_nr": unit_id,
            "time_cycles": cycle,
            "s_11": s_11,
            "s_12": s_12,
            "RUL": rul
        })

    return data


# Generate 100 historical engines
all_data = []
print("Generating training data based on 'generate_data.py' logic...")
for i in range(1, 101):
    all_data.extend(simulate_engine_life(i))

df = pd.DataFrame(all_data)
df.to_csv("training_data.csv", index=False)
print(f"Done. Saved {len(df)} rows to 'training_data.csv'.")