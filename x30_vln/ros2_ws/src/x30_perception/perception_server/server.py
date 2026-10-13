"""Entry point for docker/images/perception/entrypoint.sh:  uvicorn server:app --app-dir <this directory>.

The directory is mounted as /app/perception_server, so its parent is put on sys.path and the code is imported as the
package `perception_server` (relative imports inside it keep working).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception_server.app import app  # noqa: E402,F401
