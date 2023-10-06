import logging


def setup_logging(level=None):
    if level is None:
        level = logging.INFO
    logging.basicConfig(level=level)

# Run the setup_logging once when the module is imported
setup_logging()

def get_logger(name=None):
    if name is None:
        name = __name__
    return logging.getLogger(name)