"""
Vercel Python Serverless Entrypoint
------------------------------------
Vercel looks for a callable named `app` (or `handler`) in api/index.py.
We simply re-export the FastAPI app from the backend package.

The working directory on Vercel is the repo root, so we add `backend/`
to sys.path so that `from app.main import app` resolves correctly.
"""

import sys
from pathlib import Path

# Make `backend/` importable
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.main import app  # noqa: E402  (import after sys.path manipulation)

# Vercel expects the ASGI app to be named `app`
__all__ = ["app"]
