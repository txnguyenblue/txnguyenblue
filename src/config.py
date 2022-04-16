from pathlib import Path
from logzero import setup_logger

import logging

logger = setup_logger(__file__, level=logging.DEBUG)
file_dir = Path(__file__).resolve().parent
base_dir = file_dir.parent

class CONFIG:
    data = base_dir / "data"
    src = base_dir / "src"
    notebooks = base_dir / "notebooks"
    reports = base_dir / "reports"
    tests = base_dir / "tests"
    utils = base_dir / "utils"
if __name__ == "__main__":
    logger.info(f"Current base dir: {base_dir}")
