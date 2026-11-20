"""Check saved Panda PickPlace actions against saved MuJoCo states.

Read-only: loads a processed episode and its matching raw model.xml, steps
recorded actions in a new headless environment, and writes a JSON report.
This does NOT generate augmented demonstrations.

Usage:
    python3 validate_action_replay.py --steps 300
    python3 validate_action_replay.py --steps 0        # entire episode

Requires inspect_mimicgen_readiness.py alongside this script.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from inspect_mimicgen_readiness import (
    build_environment,
    find_raw_source,
    load_source_model,
)


def get_live_poses(env):
    """Read poses from the current simulator state without changing it."""
    robot = env.robots[0]
    eef_site_ids = robot.eef_site_id
    site_id = (eef_site_ids["right"] if "right" in eef_site_ids
               else next(iter(eef_site_ids.values()))) if isinstance(eef_site_ids, dict) else eef_site_ids
    eef_pos = np.array(env.sim.data.site_xpos[site_id], dtype=float)
    obj_pos = {
        obj.name: np.array(env.sim.data.body_xpos[env.obj_body_id[obj.name]], dtype=float)
        for obj in env.objects
    }
    return eef_pos, obj_pos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", default="processed_dataset/episode_001")
    parser.add_argument("--raw-root", default="rgb_demonstrations_raw")
    parser.add_argument("--steps", type=int, default=300,
                        help="number of actions to test; 0 means all")
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--output", default="mimicgen_prototype/action_replay_report.json")
    args = parser.parse_args()

    episode = Path(args.episode)
    states = np.load(episode / "states.npy", mmap_mode="r")
    actions = np.load(episode / "actions.npy", mmap_mode="r")
    if states.ndim != 2 or actions.ndim != 2 or len(states) != len(actions) + 1:
        raise ValueError("Invalid shapes: need states (N+1,D), actions (N,A)")
    if args.steps < 0 or args.report_every <= 0:
        raise ValueError("--steps must be >= 0; --report-every must be > 0")
    n_steps = len(actions) if args.steps == 0 else min(args.steps, len(actions))
    if n_steps == 0:
        raise ValueError("Episode has no actions")

    matches = find_raw_source(Path(args.raw_root), np.asarray(states[0]))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one matching raw recording; found {len(matches)}")
    xml_files = list(matches[0].glob("*.xml"))
    if len(xml_files) != 1:
        raise RuntimeError(f"Expected one XML in {matches[0]}, found {len(xml_files)}")

    print("=== Replay validation ===")
    print("Episode:", episode)
    print("Original XML:", xml_files[0])
    print(f"Actions to replay: {n_steps} / {len(actions)}")
    print("Original files are never modified.\n")

    env = None
    reference_env = None
    try:
        env = build_environment()
        load_source_model(env, xml_files[0])
        reference_env = build_environment()
        load_source_model(reference_env, xml_files[0])

        sim_state_dim = len(env.sim.get_state().flatten())
        lo, hi = env.action_spec
        if states.shape[1] != sim_state_dim:
            raise RuntimeError(f"Simulator state dim {sim_state_dim} != saved {states.shape[1]}")
        if actions.shape[1] != len(lo):
            raise RuntimeError(f"Action dim {len(lo)} != saved {actions.shape[1]}")
        if not np.all(np.isfinite(states[:n_steps + 1])) or not np.all(np.isfinite(actions[:n_steps])):
            raise RuntimeError("Nonfinite state or action encountered")
        bounds_ok = np.all((actions[:n_steps] >= lo - 1e-6) & (actions[:n_steps] <= hi + 1e-6))
        if not bounds_ok:
            raise RuntimeError("Recorded actions outside environment limits; will not alter them")

        env.sim.set_state_from_flattened(np.asarray(states[0], dtype=float))
        env.sim.forward()
        state0 = env.sim.get_state().flatten()
        print(f"Initial state max |diff|: {np.max(np.abs(state0 - states[0])):.3e}")

        nq = int(env.sim.model.nq)
        max_state_error = 0.0
        max_qpos_error = 0.0
        first_qpos_over_1e3 = None
        checkpoints = []
        print("step | max state error | max qpos error | EEF position drift (m) | max object drift (m)")
        print("-" * 91)
        for j in range(n_steps):
            env.step(np.asarray(actions[j], dtype=float))
            observed = env.sim.get_state().flatten()
            expected = np.asarray(states[j + 1], dtype=float)
            state_err = float(np.max(np.abs(observed - expected)))
            # Flattened MjSimState layout: [time, qpos..., qvel..., act...].
            qpos_err = float(np.max(np.abs(observed[1:1 + nq] - expected[1:1 + nq])))
            max_state_error = max(max_state_error, state_err)
            max_qpos_error = max(max_qpos_error, qpos_err)
            if first_qpos_over_1e3 is None and qpos_err > 1e-3:
                first_qpos_over_1e3 = j + 1

            if (j == 0 or (j + 1) % args.report_every == 0 or j + 1 == n_steps):
                reference_env.sim.set_state_from_flattened(expected)
                reference_env.sim.forward()
                live_eef, live_objs = get_live_poses(env)
                ref_eef, ref_objs = get_live_poses(reference_env)
                eef_drift = float(np.linalg.norm(live_eef - ref_eef))
                obj_drifts = {
                    name: float(np.linalg.norm(live_objs[name] - ref_objs[name]))
                    for name in ref_objs
                }
                max_obj_drift = max(obj_drifts.values(), default=0.0)
                print(f"{j + 1:4d} | {state_err:15.4e} | {qpos_err:14.4e} |"
                      f" {eef_drift:22.4e} | {max_obj_drift:20.4e}", flush=True)
                checkpoints.append({
                    "step": j + 1,
                    "max_absolute_state_error": state_err,
                    "max_absolute_qpos_error": qpos_err,
                    "eef_position_drift_m": eef_drift,
                    "object_position_drift_m": obj_drifts,
                })

        whole_episode = n_steps == len(actions)
        # Evaluation on only a short prefix is not meaningful for task success.
        task_success_end = bool(env._check_success()) if whole_episode else None
        report = {
            "episode": str(episode),
            "source_xml": str(xml_files[0]),
            "num_actions_total": int(len(actions)),
            "num_actions_replayed": int(n_steps),
            "full_episode": whole_episode,
            "max_absolute_state_error": max_state_error,
            "max_absolute_qpos_error": max_qpos_error,
            "first_step_qpos_error_over_1e-3": first_qpos_over_1e3,
            "success_at_final_replay_state_if_full": task_success_end,
            "checkpoints": checkpoints,
            "warning": "Action replay can drift. This script neither repairs trajectories nor generates data.",
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("\nSaved:", output)
        print(f"Max qpos drift: {max_qpos_error:.3e}")
        if first_qpos_over_1e3 is not None:
            print(f"NOTE: Joint positions departed by >1e-3 at step {first_qpos_over_1e3}.")
            print("This is not a replay PASS. Inspect controller alignment before augmentation.")
        else:
            print("Good: qpos drift stayed <=1e-3 for tested actions (not yet augmentation validation).")
        if whole_episode:
            print("Task success at final replay state:", task_success_end)
        else:
            print("Partial replay only; no end-to-end task success conclusion.")
    finally:
        if reference_env is not None:
            reference_env.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

