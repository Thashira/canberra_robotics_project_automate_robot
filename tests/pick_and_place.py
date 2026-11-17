#intial test file for scripting and see observation happening.
import numpy as np
import robosuite as suite

# Create the simulation environment
env = suite.make(
    env_name="PickPlace",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    control_freq=20,
    horizon=500,
    reward_shaping=False,
)

# Reset environment
obs = env.reset()

# Inspect observations
print("\n--- Observation Information ---")

for key, value in obs.items():
    if isinstance(value, np.ndarray):
        print(f"{key}: shape = {value.shape}")
    else:
        print(f"{key}: {value}")

# Inspect action space
print("\n--- Action Information ---")
print("Action dimension:", env.action_dim)
print("Action specification:", env.action_spec)

# Run simulation with zero actions
print("\nStarting simulation...")

for step in range(200):
    action = np.zeros(env.action_dim)

    obs, reward, done, info = env.step(action)

    env.render()

print("Simulation completed!")

env.close()
