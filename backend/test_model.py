import sys
from pathlib import Path

import numpy as np

# Allow importing model.py from this folder
sys.path.append(str(Path(__file__).resolve().parent))

from model import predict, DEPTHS


# Load an existing validation sample
PROJECT_DIR = Path(__file__).resolve().parent.parent

X_val = np.load(PROJECT_DIR / "X_val.npy")

print("X_val shape:", X_val.shape)

# Take first validation day
sample = X_val[0]

print("Sample shape:", sample.shape)

# Replace any remaining NaNs in input
sample = np.nan_to_num(sample, nan=0.0)

# Run model
prediction = predict(sample)

print("Prediction shape:", prediction.shape)

print("\nDepth-wise mean predicted temperature:")

for depth, temp_map in zip(DEPTHS, prediction):
    print(
        f"{depth:6.1f} m : "
        f"mean={np.mean(temp_map):.2f}°C | "
        f"min={np.min(temp_map):.2f}°C | "
        f"max={np.max(temp_map):.2f}°C"
    )