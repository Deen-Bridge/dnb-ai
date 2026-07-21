"""Test configuration for importing repository modules without packaging the app."""

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Fake key so modules that check at import time (main.py) don't fail
os.environ.setdefault("GEMINI_API_KEY", "test-fake-key")
