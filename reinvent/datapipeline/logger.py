"""Logging setup for single- and multi-processing"""

__all__ = ["setup_sp_logger", "logging_listener"]
import sys
import logging
import logging.handlers
from typing import TextIO

FORMATTER = logging.Formatter(
    fmt="%(asctime)s <%(levelname)-4.4s> %(message)s",
    datefmt="%H:%M:%S",
)


def setup_sp_logger(
    filename: str = None,
    name: str = __name__,
    stream: TextIO = sys.stderr,
    propagate: bool = True,
    level=logging.INFO,
):
    """Setup simple single-processing logging

    :param filename: optional filename for logging output
    :param stream: the output stream
    :param propagate: whether to propagate to higher level loggers
    :param level: logging level
    :returns: the newly set up logger
    """

    logging.captureWarnings(True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if filename:
        handler = logging.FileHandler(filename, mode="w")
    else:
        handler = logging.StreamHandler(stream)

    handler.setFormatter(FORMATTER)
    handler.setLevel(level)

    logger.addHandler(handler)
    logger.propagate = propagate

    return logger


def logging_listener(queue, filename: str, name: str = __name__, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.FileHandler(filename, mode="w+")
    handler.setFormatter(FORMATTER)
    handler.setLevel(level)

    logger.addHandler(handler)
    logger.propagate = False

    while True:
        record = queue.get()

        if record is None:
            break

        logger = logging.getLogger(record.name)
        logger.handle(record)


def setup_mp_logger(logger, level, queue):
    logger.setLevel(level)

    handler = logging.handlers.QueueHandler(queue)
    logger.addHandler(handler)
