"""A simple reporter facilituy to write information to a remote host

The functionality is somewhat reminiscent of Python logging but different
enough to call it a "reporter" rather than a "logger".  The code is also very
simplistic and only supports the task at hand.

The caller is expected to set up the logger before use.  If this does not
happen sending to the reporter will still be possible but will have no effect
i.e. a reporter will always be available.
"""

__all__ = ["setup_reporter", "get_reporter"]
import requests
import json
import logging
from typing import Mapping, Optional

logger = logging.getLogger(__name__)


HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": None,
}

MAX_ERR_MSG = 5


class RemoteJSONReporter:
    """Simplistic reporter that sends JSON to a remote server"""

    def __init__(self, url, token=None):
        """Set up the reporter

        :param url: URL to send JSON to
        :param token: access token for the URL
        """

        self.url = url

        if token:
            self.headers["Authorization"] = token

        self.headers = HEADERS
        self.max_msg = 0

    def send(self, record) -> None:
        """Send a record to a remote URL

        :param record: dictionary-like record to send to remote URL
        """

        if not isinstance(record, Mapping):
            raise TypeError("The record is expected to be a mapping")

        json_msg = json.dumps(record, indent=2)

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
