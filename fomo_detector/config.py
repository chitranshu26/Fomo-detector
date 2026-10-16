from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "fomo_realistic_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fomo_model.pkl"
LOG_PATH = PROJECT_ROOT / "logs" / "simulation_log.csv"

FEATURE_COLUMNS = (
    "alpha",
    "beta",
    "theta",
    "gamma",
    "attention",
    "scroll_speed",
    "fomo_score",
)
LABEL_COLUMN = "label"
TIMESTAMP_COLUMN = "timestamp"