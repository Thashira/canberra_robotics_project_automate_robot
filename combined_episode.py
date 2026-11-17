import os
import glob
import json
import numpy as np
import sys

raw_folder = sys.argv[1]
output_folder = sys.argv[2]

os.makedirs(output_folder, exist_ok=True)

# Find all chunks
files = glob.glob(os.path.join(raw_folder, "state_*.npz"))

# Sort chunks using their actual simulation timestep
chunks = []

for file in files:
    data = np.load(file, allow_pickle=True)
    first_step = int(data["camera_steps"][0])
    chunks.append((first_step, file))

chunks.sort(key=lambda x: x[0])

all_states = []
all_actions = []
all_images = []
all_steps = []

for _, file in chunks:
    data = np.load(file, allow_pickle=True)

    all_states.extend(data["states"])
    all_images.extend(data["camera_images"])
    all_steps.extend(data["camera_steps"])

    # Extract action vectors from action dictionaries
    for action_info in data["action_infos"]:
        all_actions.append(action_info["actions"])

# Convert lists into NumPy arrays
states = np.asarray(all_states)
actions = np.asarray(all_actions)
images = np.asarray(all_images, dtype=np.uint8)
steps = np.asarray(all_steps)

# Validate data alignment
assert len(states) == len(images)
assert len(states) == len(actions) + 1
assert np.array_equal(steps, np.arange(len(steps)))

# Save arrays
np.save(os.path.join(output_folder, "states.npy"), states)
np.save(os.path.join(output_folder, "actions.npy"), actions)

# Save RGB images in compressed format
np.savez_compressed(
    os.path.join(output_folder, "images.npz"),
    images=images
)

# Episode metadata
metadata = {
    "episode": "episode_001",
    "task": "PickPlace",
    "robot": "Panda",
    "num_states": len(states),
    "num_actions": len(actions),
    "num_images": len(images),
    "image_shape": list(images.shape[1:]),
    "state_dimension": states.shape[1],
    "action_dimension": actions.shape[1],
    "successful": bool(
        np.load(chunks[-1][1], allow_pickle=True)["successful"]
    )
}

with open(os.path.join(output_folder, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("Dataset conversion completed!")
print("States:", states.shape)
print("Actions:", actions.shape)
print("Images:", images.shape)
print("Success:", metadata["successful"])