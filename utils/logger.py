import logging
from logzero import setup_logger
from setuptools import setup

LOGGER = setup_logger(name = "ML Educational Program", level=logging.DEBUG)

if __name__ == "__main__":
    LOGGER.info("Utilities for machine learning educational program")