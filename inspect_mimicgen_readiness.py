"""Read-only readiness check for Panda PickPlace MimicGen-style augmentation.

Run from ~/Documents/canberra_robotics using the existing Python environment.
This script does not modify original demonstration data.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config


def find_raw_source(raw_root: Path, first_state: np.ndarray):
    """Match the processed episode against a raw episode by its t=0 state."""
    matches = []
    if not raw_root.is_dir():
        return matches
    for directory in sorted(raw_root.glob("ep_*")):
        if not directory.is_dir():
            continue
        for chunk in sorted(directory.glob("state_*.npz")):
            try:
                with np.load(chunk, allow_pickle=True) as data:
                    steps = data["camera_steps"]
                    if len(steps) == 0 or int(steps[0]) != 0:
                        continue
                    initial = np.asarray(data["states"][0], dtype=np.float64)
                if initial.shape == first_state.shape and np.allclose(
                    initial, first_state, atol=1e-9, rtol=1e-9
                ):
                    matches.append(directory)
                break  # first t=0 chunk is enough for this raw episode
            except (OSError, KeyError, ValueError, IndexError) as ex:
                print(f"[WARN] Could not inspect {chunk}: {ex}")
    return matches


def build_environment():
    # Same task, robot, controller, control frequency, and reward settings
    # as the original collector. No rendering is needed to inspect poses.
    config = load_composite_controller_config(controller=None, robot="Panda")
    return suite.make(
        env_name="PickPlace",
        robots=["Panda"],
        controller_configs=config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        reward_shaping=True,
        control_freq=20,
    )


def load_source_model(env, xml_path: Path):
    xml = xml_path.read_text(encoding="utf-8")
    env.reset()
    # robosuite's own demonstration playback also restores the episode XML.
    xml = env.edit_model_xml(xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.forward()


def inspect_pose(env, flat_state):
    env.sim.set_state_from_flattened(np.asarray(flat_state, dtype=np.float64))
    env.sim.forward()
    robot = env.robots[0]
    eef_ids = robot.eef_site_id
    if isinstance(eef_ids, dict):
        if not eef_ids:
            raise RuntimeError("robot.eef_site_id is empty")
        arm = next(iter(eef_ids))
        site_id = eef_ids[arm]
    else:
        arm, site_id = "arm", eef_ids

    eef_pos = np.asarray(env.sim.data.site_xpos[site_id]).copy()
    eef_rot = np.asarray(env.sim.data.site_xmat[site_id]).reshape(3, 3).copy()
    objects = {}
    for obj in env.objects:
        body_id = env.obj_body_id[obj.name]
        objects[obj.name] = {
            "position": np.asarray(env.sim.data.body_xpos[body_id]).round(6).tolist(),
            "quaternion_wxyz": np.asarray(env.sim.data.body_xquat[body_id]).round(6).tolist(),
        }
    return {
        "arm": str(arm),
        "eef_position": eef_pos.round(6).tolist(),
        "eef_rotation_matrix": eef_rot.round(6).tolist(),
        "objects": objects,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", default="processed_dataset/episode_001")
    parser.add_argument("--raw-root", default="rgb_demonstrations_raw")
    parser.add_argument("--output", default="mimicgen_prototype/pose_readiness.json")
    args = parser.parse_args()

    episode = Path(args.episode)
    states = np.load(episode / "states.npy", mmap_mode="r")
    actions = np.load(episode / "actions.npy", mmap_mode="r")
    with (episode / "metadata.json").open(encoding="utf-8") as handle:
        meta = json.load(handle)

    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError("Expected 2D state and action arrays")
    if len(states) != len(actions) + 1:
        raise ValueError("Alignment failed: expected states = actions + 1")

    print("=== Dataset summary ===")
    print("Episode directory:", episode)
    print("States:", states.shape, "Actions:", actions.shape)
    print("Metadata success:", meta.get("successful"), "(not independently verified)")

    matches = find_raw_source(Path(args.raw_root), np.asarray(states[0]))
    print("\n=== Matching original raw recording ===")
    for match in matches:
        print("Matched:", match)
    if len(matches) != 1:
        raise RuntimeError(
            "Need exactly one matching raw episode. Cannot safely infer "
            "the original model XML from a different recording. "
            "Check --raw-root; matches=" + str(len(matches))
        )

    raw_source = matches[0]
    xml_files = sorted(raw_source.glob("*.xml"))
    print("XML files:", [p.name for p in xml_files])
    if len(xml_files) != 1:
        raise RuntimeError(
            "Expected a single simulation XML in the matching raw episode. "
            "Inspect this folder manually: " + str(raw_source)
        )

    env = None
    try:
        env = build_environment()
        load_source_model(env, xml_files[0])
        sim_shape = env.sim.get_state().flatten().shape
        if sim_shape != states[0].shape:
            raise RuntimeError(
                f"Simulator state {sim_shape} != saved state {states[0].shape}. "
                "The environment/model is not compatible with this recording."
            )

        n = len(states)
        sample_steps = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
        readings = {}
        print("\n=== Pose inspection (restoring saved states, not stepping actions) ===")
        for step in sample_steps:
            pose = inspect_pose(env, states[step])
            readings[str(step)] = pose
            print(f"t={step:5d} EEF: {pose['eef_position']}")
            for obj_name, object_pose in pose["objects"].items():
                print(f"    {obj_name}: {object_pose['position']}")

        print("\n=== Environment summary ===")
        print("Robot arms:", env.robots[0].arms)
        print("Object names:", list(readings["0"]["objects"]))
        print("Source XML:", xml_files[0])

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "source_episode": str(episode),
            "raw_source": str(raw_source),
            "source_xml": str(xml_files[0]),
            "state_shape": list(states.shape),
            "action_shape": list(actions.shape),
            "sampled_poses": readings,
            "notice": "Pose extraction only; no action replay or MimicGen generation validated.",
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("\nSaved diagnostic report:", output_path)
        print("PASS: Source model + saved states were readable and poses were extracted.")
        print("NEXT: Validate action replay, subtask detection, and modified trajectories.")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
