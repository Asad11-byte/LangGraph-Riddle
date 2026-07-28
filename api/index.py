"""
Vercel Python (ASGI) entry point.

Vercel's Python runtime looks for an ASGI/WSGI-compatible `app` object in
this file when `api/index.py` is routed to via vercel.json. We simply
re-export the FastAPI app defined in app/main.py.
"""
from app.main import app  # noqa: F401
