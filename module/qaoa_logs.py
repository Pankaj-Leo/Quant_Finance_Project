import pandas as pd
import numpy as np
import os
import time

# Fake QAOA log
num_iterations = 10

log = []
for i in range(num_iterations):
    log.append({
        "iteration": i,
        "objective": np.random.uniform(-1, 1),  # fake objective
        "runtime": np.random.uniform(0.1, 1.0),  # fake runtime in seconds
        "shots": 1024,
        "fidelity": np.random.uniform(0.9, 1.0)
    })

qaoa_df = pd.DataFrame(log)

# Save to parquet
os.makedirs("data/processed", exist_ok=True)
qaoa_df.to_parquet("data/processed/qaoa_logs.parquet")

print(" QAOA logs saved successfully!")
