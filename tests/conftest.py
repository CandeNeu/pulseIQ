"""Make the Streamlit app's helper modules importable from the tests.

`streamlit run frontend/app.py` puts `frontend/` on sys.path, so the app does
`import ppg`. The tests need the same path.
"""

import pathlib
import sys

FRONTEND_DIR = pathlib.Path(__file__).resolve().parents[1] / "frontend"
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))
