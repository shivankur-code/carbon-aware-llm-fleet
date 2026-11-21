import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CONFIGS = ROOT / "configs"
__all__ = ["ROOT", "CONFIGS"]
