"""Transform base class

Registers every child class
"""

__all__ = ["Transform", "get_transform"]

from abc import ABC, abstractmethod


registry = {}


def get_transform(name):
    clean_name = name.lower().replace("_", "")
    return registry[clean_name]


class Transform(ABC):
    def __init__(self, params):
        self.params = params

    def __init_subclass__(cls, param_cls, **kwargs):
        """Register the subclass and its associated parameter class

        The registry key is the lower case subclass name. Note that this method
        is triggered at creation time of the class.

        :param param_cls: the parameter class to be registered
        """

        super().__init_subclass__(**kwargs)

        registry_name = cls.__name__.lower()
        registry[registry_name] = cls, param_cls

    @abstractmethod
    def __call__(self, predictions):
        ...
