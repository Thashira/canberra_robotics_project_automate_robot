import os 
import numpy as np
import time

from robosuite.wrappers import DataCollectionWrapper
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from robosuite.devices import Keyboard

from robosuite.scripts.collect_human_demonstrations import collect_human_trajectory
import json
import subprocess
import sys

from robotics.paths import PROCESSED_DEMONSTRATIONS, PROJECT_ROOT, RAW_DEMONSTRATIONS

class RGBDataCollectionWrapper(DataCollectionWrapper):
    """
    Extends robosuite's DataCollectionWrapper to collect
    RGB camera observations together with states and actions.
    """

    def __init__(self, env, directory, camera_key="agentview_image",collect_freq=1, flush_freq=100):

        super().__init__(
            env,
            directory,
            collect_freq=collect_freq,
            flush_freq=flush_freq
        )

        self.camera_key = camera_key

        # Stores RGB frames and their simulation timestep numbers
        self.camera_images = []
        self.camera_steps = []

        # Prevent parent step() from flushing before the
        # corresponding camera image has been recorded
        self._inside_step = False
        self._flush_pending = False

        # Tracks success across the entire episode
        self.episode_successful = False

        # Tracks total episode data
        self.episode_num_actions = 0
        self.episode_num_images = 0

    def reset(self):
        """
        Reset the environment and capture the initial camera image.
        """

        obs = super().reset()
        
        # Reset episode-level tracking
        self.episode_successful = False
        self.episode_num_actions = 0

        # Initial observation corresponds to timestep 0
        self.camera_images = [
            np.asarray(obs[self.camera_key], dtype=np.uint8).copy()
        ]
        

        self.camera_steps = [0]

        self.episode_num_images = 1

        return obs

    def step(self, action):
        """
        Execute one action and record its resulting RGB image.
        """

        self._inside_step = True

        try:
            # Reuse the original robosuite step implementation
            ret = super().step(action)

        finally:
            self._inside_step = False
 
     # Update episode-level success status
        if self.successful:
            self.episode_successful = True

        obs = ret[0]

        # Capture the image after executing the action
        image = np.asarray(
            obs[self.camera_key],
            dtype=np.uint8
        ).copy()


        self.camera_images.append(image)
        self.camera_steps.append(self.t)

        # Update episode-level counters
        self.episode_num_actions += 1
        self.episode_num_images += 1

        # Complete a pending flush only after saving this image
        if self._flush_pending:
            self._flush()

        return ret

    def _flush(self):
        """
        Save states, actions and RGB images together.
        """

        # Parent step() may request a flush before returning
        # its observation. Delay it until the image is captured.
        if self._inside_step:
            self._flush_pending = True
            return

        if not self.has_interaction:
            return

        if len(self.states) == 0:
            return

        # Create a file for the current data chunk
        t1, t2 = str(time.time()).split(".")

        state_path = os.path.join(
            self.ep_directory,
            f"state_{t1}_{t2}.npz"
        )

        # Get environment name
        if hasattr(self.env, "unwrapped"):
            env_name = self.env.unwrapped.__class__.__name__
        else:
            env_name = self.env.__class__.__name__

        # Save all collected information together
        np.savez_compressed(
            state_path,
            states=np.asarray(self.states),
            action_infos=self.action_infos,
            camera_images=np.asarray(
                self.camera_images,
                dtype=np.uint8
            ),
            camera_steps=np.asarray(self.camera_steps),
            successful=self.successful,
            episode_successful=self.episode_successful,
            env=env_name
        )

        print(
            f"Saved chunk: {len(self.states)} states, "
            f"{len(self.action_infos)} actions, "
            f"{len(self.camera_images)} images"
        )

        # Clear buffers after saving
        self.states = []
        self.action_infos = []
        self.camera_images = []
        self.camera_steps = []

        self.successful = False
        self._flush_pending = False


def create_environment():
    controller_config = load_composite_controller_config(
        controller=None,
        robot="Panda"
    )

    config = {
        "env_name": "PickPlace",
        "robots": ["Panda"],
        "controller_configs": controller_config
    }

    env = suite.make(
        **config,
        has_renderer=True,
        renderer="mjviewer",
        has_offscreen_renderer=True,
        render_camera="agentview",
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=128,
        camera_widths=128,
        ignore_done=True,
        reward_shaping=True,
        control_freq=20
    )

    env = VisualizationWrapper(env)

    data_directory = str(RAW_DEMONSTRATIONS)
    os.makedirs(data_directory, exist_ok=True)

    env = RGBDataCollectionWrapper(
        env,
        directory=data_directory,
        camera_key="agentview_image",
        collect_freq=1,
        flush_freq=100
    )

    return env


def collect_one_episode():

    # 1. Create a fresh environment
    env = create_environment()

    # 2. Create keyboard controller
    device = Keyboard(
        env=env,
        pos_sensitivity=1.0,
        rot_sensitivity=1.0
    )

    print("\nStarting a new episode...")
    print("Complete the PickPlace task using the keyboard.")

    # 3. Collect demonstration
    collect_human_trajectory(
        env=env,
        device=device,
        arm="right",
        max_fr=20,
        goal_update_mode="target"
    )

    # 4. Save episode metadata
    metadata = {
        "task": "PickPlace",
        "robot": "Panda",
        "episode_successful": env.episode_successful,
        "num_actions": env.episode_num_actions,
        "num_images": env.episode_num_images,
        "image_height": 128,
        "image_width": 128,
        "camera": "agentview"
    }

    metadata_path = os.path.join(
        env.ep_directory,
        "episode_metadata.json"
    )

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nEpisode collection completed.")
    print("Success:", env.episode_successful)
    print("Actions:", env.episode_num_actions)
    print("Images:", env.episode_num_images)

    # 5. Find the next available episode number
    processed_directory = str(PROCESSED_DEMONSTRATIONS)
    os.makedirs(processed_directory, exist_ok=True)

    episode_number = 1

    while os.path.exists(
        os.path.join(
            processed_directory,
            f"episode_{episode_number:03d}"
        )
    ):
        episode_number += 1

    output_folder = os.path.join(
        processed_directory,
        f"episode_{episode_number:03d}"
    )

    # 6. Convert raw chunks into one processed episode
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.process_episode",
            env.ep_directory,
            output_folder
        ],
        cwd=PROJECT_ROOT,
        check=True
    )

    print("\nProcessed episode saved to:", output_folder)
