"""Entry point for the multimodal reasoner GUI."""

from pathlib import Path

from core.model_engine import ModelEngine
from gui.chat_window import run_chat_app
from utils.rammap_runner import run_rammap_cleanup


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent

    # Run RAMMap cleanup before model load.
    run_rammap_cleanup(project_root)

    engine = ModelEngine(config_path=project_root / "config.yaml")
    run_chat_app(engine)

    # Model unload already happens on window close; run RAMMap once more afterwards.
    run_rammap_cleanup(project_root)
