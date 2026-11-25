import json
import time
from datetime import datetime
import mujoco
import numpy as np
import mujoco.viewer

from scipy.spatial.transform import Rotation

from robotics.paths import GENERATED_DATASETS, PROCESSED_DEMONSTRATIONS
from robotics.simulations.environment import (
    build_environment,
    load_source_model
)


from robotics.generation.config import SEGMENTS, SHIFTS
from robotics.generation.recording import source_model_path
from robotics.generation.trajectories import blending_weight, extract_source


SOURCE = PROCESSED_DEMONSTRATIONS / "episode_001"

def main():

    states = np.load(SOURCE / "states.npy")
    actions = np.load(SOURCE / "actions.npy")

    assert len(states) == len(actions) + 1
    assert actions.shape[1] == 7

    xml = source_model_path()

    env = build_environment()

    try:
        load_source_model(env, xml)

        robot = env.robots[0]
        site = robot.eef_site_id["right"]

        base_id = env.sim.model.body_name2id(
            robot.robot_model.root_body
        )

        print("Extracting source trajectories...")

        source_pos, source_rot = extract_source(
            env, states, site
        )

        # Restore original initial simulation state
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        # Shift all four objects
        for name, shift in SHIFTS.items():

            obj = next(
                obj for obj in env.objects
                if obj.name == name
            )

            joint = obj.joints[0]

            qpos = env.sim.data.get_joint_qpos(joint).copy()
            qpos[:3] += shift

            env.sim.data.set_joint_qpos(joint, qpos)

        env.sim.forward()

        base_R = env.sim.data.body_xmat[
            base_id
        ].reshape(3, 3).copy()

        # Prepare new recording
        new_states = [
            env.sim.get_state().flatten().copy()
        ]

        new_actions = []
        segment_results = []
        success_reached = False
        stopped_early = False

        # All original gripper transitions
        transitions = set(
            np.where(np.diff(actions[:, 6]) != 0)[0] + 1
        )

        # Live 3D viewer
        with mujoco.viewer.launch_passive(
            env.sim.model._model,
            env.sim.data._data
        ) as viewer:

            for name, start, end, grasp in SEGMENTS:

                if stopped_early:
                    break

                print(f"\nGenerating {name or 'Finish'}...")

                # Sample source EEF targets
                times = set(range(start, end + 1, 5))
                times.update([start, end])

                # Keep every grasp/release event
                times.update(
                    t for t in transitions
                    if start <= t <= end
                )

                max_error = 0.0

                for t in sorted(times):

                    if not viewer.is_running():
                        stopped_early = True
                        break

                    # Transform EEF target
                    weight = blending_weight(
                        t, grasp, end
                    )

                    shift = (
                        SHIFTS[name]
                        if name is not None
                        else np.zeros(3)
                    )

                    target_pos = (
                        source_pos[t] + weight * shift
                    )

                    target_R = source_rot[t]

                    # Reuse original gripper command
                    grip = actions[t, 6]

                    # Hold gripper commands at transitions
                    min_steps = 12 if t in transitions else 1
                    max_steps = 35 if t in transitions else 22

                    for k in range(max_steps):

                        current_pos = (
                            env.sim.data.site_xpos[site].copy()
                        )

                        current_R = (
                            env.sim.data.site_xmat[site]
                            .reshape(3, 3).copy()
                        )

                        # 3D position error
                        dp = target_pos - current_pos

                        # Orientation error
                        dr = Rotation.from_matrix(
                            target_R @ current_R.T
                        ).as_rotvec()

                        if (
                            k >= min_steps
                            and np.linalg.norm(dp) < 0.007
                            and np.linalg.norm(dr) < 0.15
                        ):
                            break

                        # Generate NEW arm action
                        action = np.zeros(env.action_dim)

                        action[:3] = np.clip(
                            0.8 * (base_R.T @ dp) / 0.05,
                            -0.65, 0.65
                        )

                        action[3:6] = np.clip(
                            0.7 * (base_R.T @ dr) / 0.5,
                            -0.65, 0.65
                        )

                        action[6] = grip

                        # Execute in simulation
                        env.step(action)

                        # Record generated data
                        new_actions.append(action.copy())

                        new_states.append(
                            env.sim.get_state().flatten().copy()
                        )

                        # Update live viewer
                        if len(new_actions) % 3 == 0:
                            viewer.sync()
                            time.sleep(0.015)

                        if len(new_actions) % 20 == 0:
                            success_reached |= bool(
                                env._check_success()
                            )

                    error = np.linalg.norm(
                        target_pos -
                        env.sim.data.site_xpos[site]
                    )

                    max_error = max(max_error, float(error))

                segment_results.append({
                    "object": name or "Finish",
                    "max_tracking_error_m": max_error,
                    "completed": not stopped_early
                })

                print(
                    f"Maximum EEF error: "
                    f"{max_error*1000:.1f} mm"
                )

            if viewer.is_running():
                viewer.sync()

        # Final result
        final_success = bool(env._check_success())

        # Save separately from original dataset
        output = GENERATED_DATASETS / (
            "full_generated_" +
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        output.mkdir(parents=True, exist_ok=False)

        np.save(
            output / "states.npy",
            np.asarray(new_states)
        )

        np.save(
            output / "actions.npy",
            np.asarray(new_actions)
        )

        metadata = {
            "source": str(SOURCE),
            "method": "mimicgen_inspired",
            "object_shifts": {
                k: v.tolist()
                for k, v in SHIFTS.items()
            },
            "num_actions": len(new_actions),
            "success_reached": bool(
                success_reached or final_success
            ),
            "final_success": final_success,
            "stopped_early": stopped_early,
            "segments": segment_results,
            "images_recorded": False,
            "qualified_for_training": False
        }

        with open(output / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print("\n===== GENERATION COMPLETED =====")
        print("New actions:", len(new_actions))
        print("Success:", metadata["success_reached"])
        print("Stopped early:", stopped_early)
        print("Saved candidate:", output)
        return output
    finally:
        env.close()
