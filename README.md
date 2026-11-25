# Canberra Robotics Research Project

The existing Panda PickPlace pipeline uses Python 3.12, MuJoCo, robosuite, and NumPy. It collects keyboard-teleoperated demonstrations with 128 × 128 RGB frames, robot states, and actions, then automatically processes each episode. The project also includes behavior cloning and MimicGen-inspired trajectory experiments. The research goal is to predict manipulation failures before execution and study disturbances and recovery behavior.

## Project layout

| Location | Contents |
| --- | --- |
| `robotics/data/` | RGB recorder and teleoperation in `collection`, and raw chunk conversion in `processing` |
| `robotics/simulations/` | PickPlace environment, recorded XML restoration, and pose inspection in `environment` |
| `robotics/generation/` | Source matching and inspection in `readiness`, saved action replay in `replay`, segment and shift settings in `config`, source trajectory extraction and blending in `trajectories`, synthesis in `generator`, and RGB reconstruction in `recording` |
| `robotics/learning/` | MLP training in `train_bc`, simulator rollout in `evaluate_bc`, and action prediction comparison in `plot_bc` |
| `robotics/paths.py` | Project-relative locations for datasets, models, and outputs |
| `scripts/` | Canonical commands for collection, processing, diagnostics, generation, and behavior cloning |
| `datasets/raw/demonstrations/` | Active destination for recorded episode chunks, model XML, and metadata |
| `datasets/processed/demonstrations/` | Numbered episodes with `states.npy`, `actions.npy`, `images.npz`, and `metadata.json` |
| `datasets/generated/` | Active destination for generated PickPlace episodes |
| `models/behavior_cloning/` | Active destination for the trained MLP and state scaler |
| `experiments/` | Reserved for active experimental implementations |
| `outputs/` | Plots, images, and JSON reports under `plots/`, `images/`, and `reports/` |
| `tests/` | Reserved for automated tests |
| `cpp/` | Empty `include/`, `src/`, `bindings/`, and `tests/` directories for future profiled C++ work |
| `benchmarks/` | Reserved for performance benchmarking |

## Run existing workflows

Run these commands from the project root with the existing environment active.

```bash
source .venv/bin/activate
MUJOCO_GL=egl python -m scripts.collect_dataset
```

Batch collection prompts between ten keyboard-teleoperated episodes. Each episode is saved and converted automatically.

| Workflow | Command |
| --- | --- |
| Standalone collection sequence | `MUJOCO_GL=egl python -m scripts.collect_dataset --standalone` |
| Convert an existing raw episode | `python -m scripts.process_episode RAW_EPISODE_DIR OUTPUT_EPISODE_DIR` |
| Inspect episode readiness | `python -m scripts.inspect_mimicgen_readiness --help` |
| Validate saved action replay | `python -m scripts.validate_action_replay --help` |
| Generate a PickPlace episode | `python -m scripts.generate_dataset` |
| Train the behavior-cloning MLP | `python -m scripts.train_model` |
| Evaluate the trained policy | `python -m scripts.evaluate_bc_policy` |
| Compare predicted actions | `python -m scripts.plot_bc_predictions` |

The standalone sequence collects two episodes without the batch prompt. Collection can populate the empty dataset directories. Conversion requires an existing raw episode. Readiness and replay require matching raw and processed episodes and default to `episode_001`. Generation additionally needs `outputs/reports/pose_readiness.json` and a source episode compatible with its existing segment indices. Its RGB stage runs in a separate EGL process.

Training needs enough compatible processed episodes for the existing train, validation, and test split. Policy evaluation needs both the trained model and scaler. The action comparison also needs processed `episode_015`.

The Week 01 archive is at `~/Documents/canberra_robotics_archive/week_01/`. See the [cleanup report](CLEANUP.md) for its contents and verification results.
