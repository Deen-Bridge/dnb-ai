"""Test configuration for importing repository modules without packaging the app."""

import os
import sys
from pathlib import Path

# Allow tests that import main.py to bypass the GEMINI_API_KEY check
os.environ.setdefault("GEMINI_API_KEY", "test-key-placeholder")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
