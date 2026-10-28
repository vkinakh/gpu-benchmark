import logging
import os


def setup_logger(log_filename: str) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    for old in logger.handlers[:]:
        old.close()
        logger.removeHandler(old)
    handler = (
        logging.FileHandler(log_filename)
        if int(os.environ.get("RANK", "0")) == 0
        else logging.NullHandler()
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
