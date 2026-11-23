"""
PredictiveX: Hugging Face Spaces & Streamlit Root Entry Point.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from frontend.dashboard import main

if __name__ == "__main__":
    main()
