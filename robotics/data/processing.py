# import os
# import glob
# import json
# import numpy as np

# def process_episode(raw_folder, output_folder):
#     os.makedirs(output_folder, exist_ok=True)

#     # Find all chunks
#     files = glob.glob(os.path.join(raw_folder, "state_*.npz"))

#     # Sort chunks using their actual simulation timestep
#     chunks = []

#     for file in files:
#         data = np.load(file, allow_pickle=True)
#         first_step = int(data["camera_steps"][0])
#         chunks.append((first_step, file))

#     chunks.sort(key=lambda x: x[0])

#     all_states = []
#     all_actions = []
#     all_images = []
#     all_steps = []

#     for _, file in chunks:
#         data = np.load(file, allow_pickle=True)

#         all_states.extend(data["states"])
#         all_images.extend(data["camera_images"])
#         all_steps.extend(data["camera_steps"])

#         # Extract action vectors from action dictionaries
#         for action_info in data["action_infos"]:
#             all_actions.append(action_info["actions"])

#     # Convert lists into NumPy arrays
#     states = np.asarray(all_states)
#     actions = np.asarray(all_actions)
#     images = np.asarray(all_images, dtype=np.uint8)
#     steps = np.asarray(all_steps)

#     # Validate data alignment
#     assert len(states) == len(images)
#     assert len(states) == len(actions) + 1
#     assert np.array_equal(steps, np.arange(len(steps)))

#     # Save arrays
#     np.save(os.path.join(output_folder, "states.npy"), states)
#     np.save(os.path.join(output_folder, "actions.npy"), actions)

#     # Save RGB images in compressed format
#     np.savez_compressed(
#         os.path.join(output_folder, "images.npz"),
#         images=images
#     )

#     metadata_path = os.path.join(raw_folder, "episode_metadata.json")

#     if os.path.exists(metadata_path):
#         with open(metadata_path) as f:
#             metadata = json.load(f)
#     else:
#         metadata = {}
    
#     # Episode metadata
#     metadata.update ({
#         "episode":os.path.basename(os.path.normpath(output_folder)),
#         "task": "PickPlace",
#         "robot": "Panda",
#         "num_states": len(states),
#         "num_actions": len(actions),
#         "num_images": len(images),
#         "image_shape": list(images.shape[1:]),
#         "state_dimension": states.shape[1],
#         "action_dimension": actions.shape[1],
#         "successful": bool(
#             np.load(chunks[-1][1], allow_pickle=True)["successful"]
#         )
#     })

#     with open(os.path.join(output_folder, "metadata.json"), "w") as f:
#         json.dump(metadata, f, indent=4)

#     print("Dataset conversion completed!")
#     print("States:", states.shape)
#     print("Actions:", actions.shape)
#     print("Images:", images.shape)
#     print("Success:", metadata["successful"])

import os
import glob
import json
import numpy as np
import robotics_cpp


def process_episode(raw_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    files = glob.glob(
        os.path.join(raw_folder, "state_*.npz")
    )

    if not files:
        raise ValueError("No raw episode chunks found")

    chunks = []

    for file in files:
        with np.load(file, allow_pickle=True) as data:
            first_step = int(data["camera_steps"][0])
            chunks.append((first_step, file))

    chunks.sort(key=lambda x: x[0])

    state_chunks = []
    action_chunks = []
    image_chunks = []
    step_chunks = []

    for _, file in chunks:

        with np.load(file, allow_pickle=True) as data:

            state_chunks.append(data["states"])
            action_chunks.append(data["action_infos"])
            image_chunks.append(data["camera_images"])
            step_chunks.append(data["camera_steps"])

    # C++ processing backend

    states = robotics_cpp.aggregate_states(state_chunks)

    actions = robotics_cpp.extract_actions(action_chunks)

    images = robotics_cpp.aggregate_images(image_chunks)

    steps = np.concatenate(step_chunks)

    del state_chunks
    del action_chunks
    del image_chunks
    del step_chunks

    # C++ validation

    if not robotics_cpp.validate_alignment(
        states, actions, images
    ):
        raise ValueError("Dataset alignment failed")

    if len(steps) != len(images):
        raise ValueError("Camera step count mismatch")

    if not robotics_cpp.validate_steps(steps):
        raise ValueError("Invalid simulation timesteps")

    # Save dataset

    np.save(
        os.path.join(output_folder, "states.npy"),
        states
    )

    np.save(
        os.path.join(output_folder, "actions.npy"),
        actions
    )

    np.savez_compressed(
        os.path.join(output_folder, "images.npz"),
        images=images
    )

    # Metadata

    metadata_path = os.path.join(
        raw_folder, "episode_metadata.json"
    )

    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    with np.load(
        chunks[-1][1], allow_pickle=True
    ) as data:

        successful = bool(data["successful"])

    metadata.update({
        "episode": os.path.basename(
            os.path.normpath(output_folder)
        ),
        "task": "PickPlace",
        "robot": "Panda",
        "num_states": len(states),
        "num_actions": len(actions),
        "num_images": len(images),
        "image_shape": list(images.shape[1:]),
        "state_dimension": states.shape[1],
        "action_dimension": actions.shape[1],
        "successful": successful
    })

    with open(
        os.path.join(output_folder, "metadata.json"),
        "w"
    ) as f:
        json.dump(metadata, f, indent=4)

    print("Dataset conversion completed!")
    print("Backend: C++")
    print("States:", states.shape)
    print("Actions:", actions.shape)
    print("Images:", images.shape)
    print("Success:", metadata["successful"])