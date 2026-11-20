import numpy as np
import pickle
import robosuite as suite

import joblib

from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

# # 1. Load trained BC model and scaler
# with open("bc_model.pkl", "rb") as f:
#     model = pickle.load(f)

model=joblib.load("bc_model.pkl")
# with open("state_scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)
scaler=joblib.load("state_scaler.pkl")
print("BC model loaded.")
print("Expected state dimensions:", scaler.n_features_in_)
print("Model action dimensions:", model.n_outputs_)

# 2. Create the same Panda PickPlace environment
controller_config = load_composite_controller_config(
    controller=None,
    robot="Panda"
)

env = suite.make(
    env_name="PickPlace",
    robots=["Panda"],
    controller_configs=controller_config,
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

try:
    # 3. Reset environment
    obs = env.reset()

    # Get the same type of state used during training
    state = env.sim.get_state().flatten()

    assert len(state) == scaler.n_features_in_, (
        f"State mismatch: got {len(state)}, "
        f"expected {scaler.n_features_in_}"
    )

    # Get valid action limits
    action_low, action_high = env.action_spec

    print("\nEnvironment ready.")
    print("Live state shape:", state.shape)
    print("Action limits:", action_low, action_high)

    # 4. Execute the learned policy
    max_steps = 10000
    success = False

    print("\nStarting BC policy execution...")

    for step in range(max_steps):

        # Scale live state using the training scaler
        state_scaled = scaler.transform(
            state.reshape(1, -1)
        )

        # Predict action
        predicted_action = model.predict(state_scaled)[0]

        # Keep action within environment limits
        action = np.clip(
            predicted_action,
            action_low,
            action_high
        )

        # Execute action
        obs, reward, done, info = env.step(action)

        # Get the next simulator state
        state = env.sim.get_state().flatten()

        if step % 50 == 0:
            print(
                f"Step: {step:4d} | "
                f"Reward: {reward:.4f} | "
                f"Action: {np.round(action, 3)}"
            )

        # Check task success
        if env._check_success():
            success = True
            print(f"\nPickPlace SUCCESS at step {step}!")
            break

    if not success:
        print("\nPolicy did not complete PickPlace.")
        print("Maximum evaluation steps reached.")

finally:
    env.close()
    print("Environment closed.")
