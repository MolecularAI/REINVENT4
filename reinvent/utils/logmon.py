"""Logging and monitoring support

Setup for Python and RDKit logging.  Setup for remote monito
"""

from __future__ import annotations

__all__ = [
    "CsvFormatter",
    "setup_logger",
    "enable_rdkit_log",
    "setup_responder",
    "setup_reporter",
    "get_reporter",
]

import json
import os
import sys
import csv
import io
import logging
from logging.config import dictConfig, fileConfig
from typing import List, Mapping, Optional

import math
import requests
from rdkit import RDLogger

logger = logging.getLogger(__name__)

HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": None,
}

MAX_ERR_MSG = 5
RESPONDER_TOKEN = "RESPONDER_TOKEN"


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


def enable_rdkit_log(levels: List[str]):
    """Enable logging messages from RDKit for a specific logging level.

    :param levels: the specific level(s) that need to be silenced
    """

    if "all" in levels:
        RDLogger.EnableLog("rdApp.*")
        return

    for level in levels:
        RDLogger.EnableLog(f"rdApp.{level}")


def setup_responder(config):
    """Setup for remote monitor

    :param config: configuration
    """

    endpoint = config.get("endpoint", False)

    if not endpoint:
        return

    token = os.environ.get(RESPONDER_TOKEN, None)
    setup_reporter(endpoint, token)


class NanInfEncoder(json.JSONEncoder):
    def _custom_encoder(self, obj):
        """Recursively clean nested dictionaries and handle NaN/Infinity"""
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None  # Return None for NaN or Infinity
        elif isinstance(obj, dict):
            # Recursively clean nested dictionaries
            return {key: self._custom_encoder(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # Recursively clean lists and tuple
            return [self._custom_encoder(item) for item in obj]
        return obj

    def encode(self, obj, *args, **kwargs):
        return super().encode(self._custom_encoder(obj), *args, **kwargs)


class RemoteJSONReporter:
    """Simplistic reporter that sends JSON to a remote server"""

    def __init__(self, url, token=None):
        """Set up the reporter

        :param url: URL to send JSON to
        :param token: access token for the URL
        """

        self.url = url
        self.headers = HEADERS

        if token:
            self.headers["Authorization"] = token

        self.max_msg = 0

    def send(self, record) -> None:
        """Send a record to a remote URL

        :param record: dictionary-like record to send to remote URL
        """

        if not isinstance(record, Mapping):
            raise TypeError("The record is expected to be a mapping")

        json_msg = json.dumps(record, cls=NanInfEncoder, indent=2)

        logger.debug(
            "Data sent to {url}\n\n{headers}\n\n{json_data}".format(
                url=self.url,
                headers="\n".join(f"{k}: {v}" for k, v in self.headers.items()),
                json_data=json_msg,
            )
        )

        response = requests.post(self.url, json=json.loads(json_msg), headers=self.headers)

        # alternative: check if response.status_code != request.codes.created
        if not response.ok and self.max_msg < MAX_ERR_MSG:
            self.max_msg += 1
            logger.error(f"Failed to send record to: {self.url}")
            logger.error(f"{response.text=}")
            logger.error(f"{response.headers=}")
            logger.error(f"{response.reason=}")
            logger.error(f"{response.url=}")


_reporter = None


def get_reporter() -> Optional[RemoteJSONReporter]:
    """Return the current reporter

    :return: reporter object
    """

    return _reporter


def setup_reporter(url, token=None) -> bool:
    """Set up the reporter

    :param url: URL to send JSON to
    :param token: access token for the URL
    :returns: whether reporter was setup successfully
    """

    global _reporter

    if url:
        # assume endpoint is readily available...
        _reporter = RemoteJSONReporter(url, token)
        return True

    return False
