"""Make the package and the sibling x30_common importable without installing (pure pytest, no ROS needed)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent.parent / 'x30_common', HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
