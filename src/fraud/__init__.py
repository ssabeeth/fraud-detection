"""Real-time card fraud detection with point-in-time features and money-based decisions."""

import os

__version__ = "0.1.0"

# MLflow prints an advertisement for its tracing skill on import; it is noise here.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
