import unittest

from dataclasses import fields

from reinvent.chemistry.standardization.filter_configuration import FilterConfiguration
from reinvent.chemistry.standardization.rdkit_standardizer import RDKitStandardizer


def classFromArgs(className, argDict):
    fieldSet = {f.name for f in fields(className) if f.init}
    filteredArgDict = {k: v for k, v in argDict.items() if k in fieldSet}
    return className(**filteredArgDict)


class BaseRDKitStandardizer(unittest.TestCase):
    def setUp(self):
        self.raw_config = None if None else self.raw_config
        if not self.raw_config:
            raise NotImplemented(
                "Please, assign value to self.raw_config in the derived test class"
            )
        config = classFromArgs(FilterConfiguration, self.raw_config)
        filter_configs = [config]
        self.standardizer = RDKitStandardizer(filter_configs)
