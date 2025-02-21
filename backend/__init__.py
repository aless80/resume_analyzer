import logging
from typing import Literal


def configure_loggers_levels(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
):
    """
    Configure the root and src loggers' levels.

    Args:
        level: Logger specification.
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    backend_logger = logging.getLogger(__name__)
    backend_logger.setLevel(level)  # backend.logging_config
