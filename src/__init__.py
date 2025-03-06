from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
MODEL_DIR = ROOT_DIR / "data" / "models"
DATA_DIR = ROOT_DIR / "data"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)