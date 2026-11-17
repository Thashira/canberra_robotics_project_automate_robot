import os
import csv
import json
import time

import numpy as np
import robosuite as suite

from pathlib import Path


# --------------------------------------------------
# 1. EXPERIMENT CONFIGURATION
# --------------------------------------------------

# Number of simulation steps to record.
# At 20 steps per second, 100 steps represent
# approximately 5 seconds of simulation time.
NUM_STEPS = 100

# Camera resolution.
CAMERA_HEIGHT = 128
CAMERA_WIDTH = 128

# Each experiment gets its own folder.
OUTPUT_DIR = Path("dataset/episode_001")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. CREATE THE ROBOT ENVIRONMENT
# --------------------------------------------------

env = suite.make(
    env_name="PickPlace",
    robots="Panda",

    # We don't need an interactive window.
    has_renderer=False,

    # We need camera images without opening a window.
    has_offscreen_renderer=True,

    # Request visual observations from the camera.
    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=CAMERA_HEIGHT,
    camera_widths=CAMERA_WIDTH,

    # Simulation control frequency.
    control_freq=20,

    # Maximum episode length.
    horizon=NUM_STEPS,
)


# --------------------------------------------------
# 3. RESET THE ENVIRONMENT
# --------------------------------------------------

# Initialize the robot and objects.
# This returns the initial observation.
obs = env.reset()


# --------------------------------------------------
# 4. PREPARE DATA STORAGE
# --------------------------------------------------

# These lists will hold observations and actions
# from each simulation step.

records = []
camera_images = []


# --------------------------------------------------
# 5. RECORD THE INITIAL OBSERVATION
# --------------------------------------------------

# Step 0 represents the initial state.
# No action has been executed yet.

camera_images.append(obs["agentview_image"].copy())

records.append({
    "step": 0,
    "timestamp": time.time(),
    "action": None,
    "reward": None,
    "done": False,

    "robot_joint_pos": obs["robot0_joint_pos"].tolist(),
    "robot_eef_pos": obs["robot0_eef_pos"].tolist(),

    "can_pos": obs["Can_pos"].tolist(),
    "milk_pos": obs["Milk_pos"].tolist(),
    "bread_pos": obs["Bread_pos"].tolist(),
})


# --------------------------------------------------
# 6. RUN THE SIMULATION AND COLLECT DATA
# --------------------------------------------------

for step in range(NUM_STEPS):

    # Create an action containing zeros.
    # This is a neutral test command, not a learned
    # or task-completing pick-and-place action.
    action = np.zeros(env.action_dim)

    # Execute the action.
    # The environment returns the updated observation,
    # reward, termination flag and additional information.
    obs, reward, done, info = env.step(action)

    # Store the camera image after the action.
    camera_images.append(obs["agentview_image"].copy())

    # Store the corresponding robot and object states.
    records.append({
        "step": step + 1,
        "timestamp": time.time(),

        # Action that produced this resulting state.
        "action": action.tolist(),

        "reward": float(reward),
        "done": bool(done),

        "robot_joint_pos": obs["robot0_joint_pos"].tolist(),
        "robot_eef_pos": obs["robot0_eef_pos"].tolist(),

        "can_pos": obs["Can_pos"].tolist(),
        "milk_pos": obs["Milk_pos"].tolist(),
        "bread_pos": obs["Bread_pos"].tolist(),
    })

    # Print progress every 20 steps.
    if (step + 1) % 20 == 0:
        print(f"Recorded {step + 1}/{NUM_STEPS} steps")

    # Stop if the environment reports termination.
    if done:
        break


# --------------------------------------------------
# 7. SAVE STRUCTURED OBSERVATIONS AND ACTIONS
# --------------------------------------------------

csv_path = OUTPUT_DIR / "observations_actions.csv"

# JSON strings allow us to store vector values
# such as joint positions inside CSV columns.
fieldnames = list(records[0].keys())

with open(csv_path, "w", newline="") as file:

    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(records)


# --------------------------------------------------
# 8. SAVE CAMERA IMAGES
# --------------------------------------------------

image_path = OUTPUT_DIR / "camera_images.npz"

# Compressed NumPy archive.
# Each image index corresponds to the same step
# index in observations_actions.csv.
np.savez_compressed(
    image_path,
    images=np.stack(camera_images)
)


# --------------------------------------------------
# 9. SAVE EXPERIMENT METADATA
# --------------------------------------------------

metadata = {
    "environment": "PickPlace",
    "robot": "Panda",
    "camera": "agentview",
    "camera_resolution": [CAMERA_HEIGHT, CAMERA_WIDTH],
    "control_frequency": 20,
    "recorded_steps": len(records) - 1,
    "action_dimension": env.action_dim,
    "action_type": "zero_action_test",
}

with open(OUTPUT_DIR / "metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)


# --------------------------------------------------
# 10. CLOSE THE ENVIRONMENT
# --------------------------------------------------

env.close()

print("\nData collection completed!")
print("Output directory:", OUTPUT_DIR)
print("Recorded observations:", len(records))
print("Recorded camera images:", len(camera_images))