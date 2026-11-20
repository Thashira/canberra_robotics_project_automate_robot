import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model and scaler
model = joblib.load("bc_model.pkl")
scaler = joblib.load("state_scaler.pkl")

# Load one unseen test episode
folder = "./processed_dataset/episode_015"

states = np.load(os.path.join(folder, "states.npy"))
actions = np.load(os.path.join(folder, "actions.npy"))

# Align states and actions
X = states[:-1]
y_actual = actions

# Normalize states using training scaler
X_scaled = scaler.transform(X)

# Predict actions
y_predicted = model.predict(X_scaled)

# Select first 300 timesteps
N = 300

# Plot all 7 action dimensions
action_names = [
    "X Translation",
    "Y Translation",
    "Z Translation",
    "Rotation X",
    "Rotation Y",
    "Rotation Z",
    "Gripper"
]

fig, axes = plt.subplots(7, 1, figsize=(14, 18))

for i in range(7):

    axes[i].plot(
        y_actual[:N, i],
        label="Actual Action",
        linewidth=1
    )

    axes[i].plot(
        y_predicted[:N, i],
        label="Predicted Action",
        linewidth=1
    )

    axes[i].set_title(action_names[i])
    axes[i].set_ylabel("Action Value")
    axes[i].legend()
    axes[i].grid(True)

axes[-1].set_xlabel("Timestep")

plt.tight_layout()
plt.savefig("bc_action_comparison.png", dpi=150)
plt.show()