"""
Vercel Python entry point.

`api/` lives at the project root, as a sibling of `backend/`, so we need to
put `backend/` on sys.path *before* importing the `app` package — otherwise
`from app.main import app` raises ModuleNotFoundError on every cold start,
the function crashes with no HTTP response at all, and the browser reports
it as a generic "Failed to fetch" (a crashed function returns no CORS
headers, so that's all the browser can tell you).
"""
import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.main import app  # noqa: E402,F401