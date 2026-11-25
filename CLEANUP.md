# Week 01 cleanup report

## Final directory tree

```text
canberra_robotics/
├── benchmarks/
├── cpp/
│   ├── bindings/
│   ├── include/
│   ├── src/
│   └── tests/
├── datasets/
│   ├── generated/
│   ├── processed/demonstrations/
│   └── raw/demonstrations/
├── experiments/
├── models/behavior_cloning/
├── outputs/
│   ├── images/
│   ├── plots/
│   └── reports/
├── robotics/
│   ├── data/
│   ├── generation/
│   ├── learning/
│   ├── simulations/
│   └── paths.py
├── scripts/
├── tests/
├── CLEANUP.md
└── README.md
```

The existing environment, repository metadata, and VS Code JSON settings are also present.

## Removed and consolidated

- Removed the nine root launchers `collect_dataset.py`, `combined_episode.py`, `custom_demonstrator.py`, `evaluate_bc_policy.py`, `generate_pickplace.py`, `inspect_mimicgen_readiness.py`, `MLP_test.py`, `plot_bc_predictions.py`, and `validate_action_replay.py`. The corresponding workflows use eight canonical commands under `scripts/`. Both collection modes share one command.
- Removed the package-level executable guards. Moved batch orchestration, processing argument dispatch, and the generator's EGL subprocess dispatch into their respective launchers. Both internal subprocesses now resolve their targets as modules from the project root and propagate child failures.
- Reused `collect_one_episode()` for both standalone episodes. The second block differed in console wording, while its environment, keyboard, capture, metadata, conversion, and numbering matched the first block. The batch command still collects ten episodes and prompts between them.
- Removed the empty `robotics/generation/controller.py`, empty `configs/` and `research/`, and the unused empty behavior-cloning and MimicGen experiment subdirectories. Removed project bytecode caches and the generated VS Code database, lock, and sidecars. Kept both IDE JSON settings and the planned C++ structure. There was one active README at the start of cleanup. The earlier copies remain in the historical snapshot.

## Archived materials

All paths below are relative to `~/Documents/canberra_robotics_archive/week_01/`.

| Archive location | Contents |
| --- | --- |
| `legacy/week_01/` | Original Python snapshot and both original READMEs |
| `datasets/raw/demonstrations/` | Raw RGB demonstration episodes |
| `datasets/processed/demonstrations/` | Processed demonstration episodes |
| `datasets/generated/` | MimicGen prototype dataset |
| `models/behavior_cloning/` | Trained model and state scaler |
| `outputs/` | Experimental plots, images, JSON reports, and the previous restructuring report |
| `tests/` | Early manual scripts, notes, two HDF5 datasets, and camera episode artifacts |
| `experiments/camera/` | Offscreen probe, camera snapshot script, and prototype episode logger |

The active dataset, model, and output directories are empty. The historical HDF5 inspector used a working-directory-relative path and is archived with its input files.

## Commands and prerequisites

Run from the project root with the existing virtual environment active.

| Command | Required inputs |
| --- | --- |
| `MUJOCO_GL=egl python -m scripts.collect_dataset` | Keyboard and configured simulation backend |
| `MUJOCO_GL=egl python -m scripts.collect_dataset --standalone` | Same setup for the two-episode sequence |
| `python -m scripts.process_episode RAW_DIR OUTPUT_DIR` | Recorded raw episode |
| `python -m scripts.inspect_mimicgen_readiness` | Matching raw and processed episode |
| `python -m scripts.validate_action_replay` | Matching raw and processed episode |
| `python -m scripts.generate_dataset` | Compatible processed source episode, readiness report, and source XML |
| `python -m scripts.train_model` | Compatible processed episodes for the existing split |
| `python -m scripts.evaluate_bc_policy` | Trained model and scaler |
| `python -m scripts.plot_bc_predictions` | Trained model, scaler, and processed episode 015 |

## Storage and verification

| Scope | Allocated storage |
| --- | --- |
| Active project before cleanup, including repository metadata and the virtual environment | 10.11 GB |
| Active project after cleanup, on the same basis | 3.11 GB |
| External Week 01 archive | 5.71 GB |

The active project shrank by about 7.00 GB. The archive accounts for about 5.71 GB of that reduction. Removing generated files, primarily the VS Code browsing database, reclaimed about 1.28 GB across both locations.

- Compared sizes, inode numbers, and SHA-256 hashes for all 1,202 archived files before and after the rename-only move. All matched.
- Passed in-memory syntax checks for all 24 active Python files. Imported all eight canonical launchers using EGL and the dummy keyboard backend.
- Verified the ten-episode and two-episode dispatch, prompt counts, subprocess module resolution, and EGL child routing with mocks. Confirmed that the active dataset tree contains no files.
- Live teleoperation, simulation, replay, generation, rendering, and training were outside this verification run.

## Remaining technical debt and profiling candidates

Generation depends on a fixed source episode and segment offsets. The training and plotting experiments also use fixed episode selections. The repository currently has no automated runtime suite for those workflows.

Profile simulator stepping, saved-state pose extraction, trajectory processing, and RGB reconstruction separately after representative datasets are available. Pose extraction and trajectory-loop code are potential C++ candidates if measurements show Python-side overhead. Verify any eventual port against recorded Python outputs.
