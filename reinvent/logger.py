__all__ = ["CsvFormatter", "setup_logger"]
import sys
import logging
import csv
import io
from logging.config import dictConfig, fileConfig


class CsvFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.output = io.StringIO()
        self.writer = csv.writer(self.output)

    def format(self, record):
        self.writer.writerow(record.msg)  # needs to be a iterable
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()


def setup_logger(
    name: str = None,
    config: dict = None,
    filename: str = None,
    formatter=None,
    stream=sys.stderr,
    cfg_filename: str = None,
    propagate: bool = True,
    level=logging.INFO,
    debug=False,
):
    """Setup a logging facility.

    :param name: name of the logger, root if empty or None
    :param config: dictionary configuration
    :param filename: optional filename for logging output
    :param formatter: a logging formatter
    :param stream: the output stream
    :param cfg_filename: filename of a logger configuration file
    :param propagate: whether to propagate to higher level loggers
    :param level: logging level
    :param debug: set special format for debugging
    :returns: the newly set up logger
    """

    logging.captureWarnings(True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if config is not None:
        dictConfig(config)
        return

    if cfg_filename is not None:
        fileConfig(cfg_filename)
        return

    if filename:
        handler = logging.FileHandler(filename, mode="w+")
    else:
        handler = logging.StreamHandler(stream)

    handler.setLevel(level)

    if debug:
        log_format = "%(asctime)s %(module)s.%(funcName)s +%(lineno)s: %(levelname)-4s %(message)s"
    else:
        log_format = "%(asctime)s <%(levelname)-4.4s> %(message)s"

    if not formatter:
        formatter = logging.Formatter(
            fmt=log_format,
            datefmt="%H:%M:%S",
        )

    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = propagate

    return logger
