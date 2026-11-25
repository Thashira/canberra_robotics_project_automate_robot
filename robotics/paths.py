from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DEMONSTRATIONS = PROJECT_ROOT / "datasets/raw/demonstrations"
PROCESSED_DEMONSTRATIONS = PROJECT_ROOT / "datasets/processed/demonstrations"
GENERATED_DATASETS = PROJECT_ROOT / "datasets/generated"
BC_MODELS = PROJECT_ROOT / "models/behavior_cloning"
PLOTS = PROJECT_ROOT / "outputs/plots"
REPORTS = PROJECT_ROOT / "outputs/reports"
