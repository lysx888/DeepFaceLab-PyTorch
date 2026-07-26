import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

INSIGHTFACE_ROOT = os.environ.get('INSIGHTFACE_HOME', str(BASE_DIR))

MODEL_ROOT = os.environ.get('INSIGHTFACE_MODEL_ROOT', str(BASE_DIR / "models"))