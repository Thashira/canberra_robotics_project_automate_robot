import json
from pathlib import Path

import mujoco
import numpy as np

from robotics.paths import RAW_DEMONSTRATIONS, REPORTS
from robotics.simulations.environment import build_environment, load_source_model


def source_model_path():
    with (REPORTS / "pose_readiness.json").open() as f:
        xml = Path(json.load(f)["source_xml"])

    # Existing reports store the pre-restructure raw directory.
    if xml.parts[:1] == ("rgb_demonstrations_raw",):
        return RAW_DEMONSTRATIONS.joinpath(*xml.parts[1:])
    return xml


def render_images(output):

    states = np.load(output / "states.npy")
    actions = np.load(output / "actions.npy")

    assert len(states) == len(actions) + 1

    xml = source_model_path()

    env = build_environment()

    try:
        load_source_model(env, xml)

        model = env.sim.model._model
        data = env.sim.data._data

        renderer = mujoco.Renderer(
            model,
            height=128,
            width=128
        )

        try:
            images = np.empty(
                (len(states), 128, 128, 3),
                dtype=np.uint8
            )

            for t, state in enumerate(states):

                env.sim.set_state_from_flattened(state)
                env.sim.forward()

                renderer.update_scene(
                    data,
                    camera="agentview"
                )

                images[t] = renderer.render()

                if t % 500 == 0:
                    print(f"RGB: {t}/{len(states)}")

        finally:
            renderer.close()

        # Save camera observations
        np.savez_compressed(
            output / "images.npz",
            images=images
        )

        # Update existing metadata
        metadata_path = output / "metadata.json"

        with open(metadata_path) as f:
            metadata = json.load(f)

        metadata["images_recorded"] = True
        metadata["num_images"] = len(images)
        metadata["image_shape"] = [128, 128, 3]

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print("\nRGB recording completed!")
        print("States:", states.shape)
        print("Actions:", actions.shape)
        print("Images:", images.shape)
    
    finally:
        env.close()
