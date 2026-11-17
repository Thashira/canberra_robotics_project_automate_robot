#This code is created at first when create pick_and_place code we get openGL error when debug it found offscreen ON cause it.
#To find exact reason and test use this code .True
# Conclusion-Backend broken for mujoco (offline environment) so manualy actiavation for backend need.
# MUJOCO_GL=egl python test_offscreen.py  solve the problem
import numpy as np
import robosuite as suite

env = suite.make(
    env_name="PickPlace",
    robots="Panda",

    has_renderer=False,
    has_offscreen_renderer=True,

    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=128,
    camera_widths=128,

    control_freq=20,
    horizon=200,
)

try:
    obs = env.reset()

    print("\nInitial camera observation:")
    print("Image shape:", obs["agentview_image"].shape)

    for step in range(100):
        action = np.zeros(env.action_dim)
        obs, reward, done, info = env.step(action)

        if step % 20 == 0:
            print(
                f"Step {step}: "
                f"Image shape = {obs['agentview_image'].shape}"
            )

    print("\nOffscreen rendering test completed!")

finally:
    env.close()