from pathlib import Path

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config


def build_environment():
    # Same task, robot, controller, control frequency, and reward settings
    # as the original collector. No rendering is needed to inspect poses.
    config = load_composite_controller_config(controller=None, robot="Panda")
    return suite.make(
        env_name="PickPlace",
        robots=["Panda"],
        controller_configs=config,
        has_renderer=True,
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
