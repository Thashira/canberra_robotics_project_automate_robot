import numpy as np
import robosuite as suite
from PIL import Image

# Create the simulation environment
env = suite.make(
    env_name="PickPlace",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=128,
    camera_widths=128,
    control_freq=20,
    horizon=100,
)

obs=env.reset()

# Check available observations
print("\nAvailable observation keys:")
print(obs.keys())

# Get RGB camera image
image = obs["agentview_image"]


print("\nCamera image shape:", image.shape)
print("Image data type:", image.dtype)

# Take one simulation step
action = np.zeros(env.action_dim)
obs, reward, done, info = env.step(action)

print("\nImage shape after one step:", obs["agentview_image"].shape)

# Save the initial camera observation as an RGB image
Image.fromarray(image).save("camera_test.png")

print("Camera image saved: camera_test.png")

env.close()