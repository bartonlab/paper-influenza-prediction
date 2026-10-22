"""Compatibility alias for the reorganized plotting primitives."""

import importlib
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_IMPLEMENTATION = importlib.import_module("influenza_figures.legacy.mplot")
sys.modules[__name__] = _IMPLEMENTATION
