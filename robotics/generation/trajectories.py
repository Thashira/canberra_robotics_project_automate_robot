import numpy as np


def extract_source(env, states, site):

    n = len(states) - 1

    positions = np.zeros((n, 3))
    rotations = np.zeros((n, 3, 3))

    for t in range(n):

        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()

        positions[t] = env.sim.data.site_xpos[site]

        rotations[t] = (
            env.sim.data.site_xmat[site].reshape(3, 3)
        )

    return positions, rotations

def blending_weight(t, grasp, release):

    if grasp is None:
        return 0.0

    fade_start = grasp + 25
    fade_end = release - 40

    if t <= fade_start:
        return 1.0

    if t >= fade_end:
        return 0.0

    u = (t - fade_start) / (fade_end - fade_start)

    return 1.0 - u*u*(3 - 2*u)
