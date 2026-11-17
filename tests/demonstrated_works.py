import h5py
import os

# Path to the collected dataset
file_path = "./demonstrations/1794820236_5765827/demo.hdf5"

# Open the dataset
with h5py.File(file_path, "r") as f:

    data = f["data"]

    # Print environment information
    print("Environment:", data.attrs["env"])

    # Count successful demonstrations
    demos = list(data.keys())
    print("Number of demonstrations:", len(demos))

    # Inspect each demonstration
    for name in demos:
        episode = data[name]

        print("\nEpisode:", name)
        print("States:", episode["states"].shape)
        print("Actions:", episode["actions"].shape)