"""Provides util functions for all src files like logging."""
import logging


def setup_logging(level=None):
    """Set up logging.

    Args:
        level : the higher the more logging.
    """
    if level is None:
        level = logging.INFO
    logging.basicConfig(level=level)


# Run the setup_logging once when the module is imported
setup_logging()


def get_logger(name=None):
    """Public module to get a logger.

    Args:
        name : Further info like __file__

    Returns:
        _type_: logger
    """
    if name is None:
        name = __name__
    return logging.getLogger(name)
