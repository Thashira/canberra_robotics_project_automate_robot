# Canberra Robotics Research Project

## Overview
A MuJoCo and robosuite-based simulation pipeline for collecting robot manipulation demonstrations. The current implementation focuses on the Panda robot performing the PickPlace task.

## Technologies
- Python 3.12
- MuJoCo
- robosuite
- NumPy

## Current Features
- Panda robot simulation using robosuite.
- Keyboard-based human teleoperation.
- RGB camera image collection (128 × 128).
- Robot state and action recording.
- Automatic processing of raw demonstration data.
- Batch collection of multiple episodes.

## Dataset Structure

```text
canberra_robotics/
├── custom_demonstrator.py
├── combined_episode.py
├── collect_dataset.py
├── rgb_demonstrations_raw/
└── processed_dataset/
    ├── episode_001/
    │   ├── states.npy
    │   ├── actions.npy
    │   ├── images.npz
    │   └── metadata.json
    └── ...
```

## Data Collection

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run the batch collection pipeline:

```bash
MUJOCO_GL=egl python3 collect_dataset.py
```

The robot is controlled manually using the keyboard. Each episode is automatically saved and processed.

## Current Progress
- MuJoCo and robosuite environment configured.
- Panda PickPlace task implemented.
- RGB, state, and action collection implemented.
- Automatic episode processing implemented.
- Initial successful demonstration episodes collected.

## Research Direction
The project aims to investigate robot manipulation under disturbances, including task failures and recovery behaviours, and develop a dataset for robot learning and evaluation.
