"""Manage IO and signal handling

A context manager to handle IO and signals.  Saves data to a file when end of
with block is reached or a signal is received.  The data is obtained through
registering a callback.  Without this callback no data will be written.
torch.save is used to pickle the data.
"""

import signal
import platform
from pathlib import Path
from typing import Callable, Dict

import torch


if platform.system() != "Windows":
    # NOTE: SIGTERM (signal 15) seems to be triggered by terminating processes.  So
    #       multiprocessing triggers the handler for every terminating child.
    SUPPORTED_SIGNALS = (signal.SIGINT, signal.SIGQUIT)
else:
    SUPPORTED_SIGNALS = (signal.SIGINT,)


class StageInterrupted(Exception):
    pass


class Handler:
    """Simple IO and signal handler

    Make sure to register a callback to actually write out data.
    """

    def __init__(self):
        """Set up the handler."""

        self._checkpoint = None
        self._callback = None
        self._out_filename = None
        self._default_handlers = []

    def __enter__(self):
        """Set the signal handler

        Catch SIGINT (Ctrl-C) and SIGQUIT (Ctrl-\).

        FIXME: need to replace this with a better signalling method
        """

        for _signal in SUPPORTED_SIGNALS:
            self._default_handlers.append((_signal, signal.getsignal(_signal)))
            signal.signal(_signal, self._signal_handler)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Save the data and reset the signal handler"""

        # FIXME: check if exc_type is really the one we want?

        for _signal in SUPPORTED_SIGNALS:
            signal.signal(_signal, signal.SIG_IGN)

        self.save()

        for _signal, _handler in self._default_handlers:
            signal.signal(_signal, _handler)

        # prevent exception from bubbling up
        # return True

    @property
    def out_filename(self):
        """Getter to obtain the output filename"""
        return self._out_filename

    @out_filename.setter
    def out_filename(self, new_out_filename):
        """Setter to set the output filename"""
        self._out_filename = Path(new_out_filename).absolute()

    @property
    def checkpoint(self):
        """Getter to return the checkpoint"""
        return self._checkpoint

    def register_callback(self, callback_function: Callable[[], Dict]) -> None:
        """Register a callback function to retrieve the data to be stored.

        :param callback_function: the function to be called in __exit__()
        """

        self._callback = callback_function

    def save(self) -> None:
        """Save the data using torch.save.

        NB: The data is obtained through the registered callback.  If this
            cannot be done, torch.save will write out an "empty" data
            structure.
        """

        data = None

        if callable(self._callback):
            data = self._callback()

        if data:
            torch.save(data, self._out_filename)

    def _signal_handler(self, signum, frame) -> None:
        """Simple signal handler.

        Relays the signal via an exception.
            __exit__() will be called after the exception has been raised
            the caller needs to handle the exception

        :param signum: the number of the signal
        :type signum: integer
        :param frame: the frame object
        :type frame: Frame
        :raises: StageInterrrupted
        """

        raise StageInterrupted
