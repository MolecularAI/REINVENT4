import gzip
from typing import List

from reinvent.chemistry import conversions
from reinvent.chemistry.standardization.filter_configuration import FilterConfiguration
from reinvent.chemistry.standardization.rdkit_standardizer import RDKitStandardizer


class FileReader:
    def __init__(self, configuration: List[FilterConfiguration], logger):
        self._standardizer = RDKitStandardizer(configuration, logger)

    def read_library_design_data_file(
        self, file_path, ignore_invalid=True, num=-1, num_fields=0
    ) -> str:
        """
        Reads a library design data file.
        :param num_fields: Number columns from the beginning to be loaded.
        :param file_path: Path to a SMILES file.
        :param ignore_invalid: Ignores invalid lines (empty lines)
        :param num: Parse up to num rows.
        :return: An iterator with the rows.
        """

        with self._open_file(file_path, "rt") as csv_file:
            for i, row in enumerate(csv_file):
                if i == num:
                    break
                splitted_row = row.rstrip().replace(",", " ").replace("\t", " ").split()
                if splitted_row:
                    if num_fields > 0:
                        splitted_row = splitted_row[0:num_fields]
                    yield splitted_row
                elif not ignore_invalid:
                    yield None

    def _open_file(self, path, mode="r", with_gzip=False):
        """
        Opens a file depending on whether it has or not gzip.
        :param path: Path where the file is located.
        :param mode: Mode to open the file.
        :param with_gzip: Open as a gzip file anyway.
        """
        open_func = open
        if path.endswith(".gz") or with_gzip:
            open_func = gzip.open
        return open_func(path, mode)

    def read_delimited_file(
        self, file_path, ignore_invalid=True, num=-1, standardize=False, randomize=False
    ):
        """
        Reads a file with SMILES strings in the first column.
        :param randomize: Standardizes smiles.
        :param standardize: Randomizes smiles.
        :param file_path: Path to a SMILES file.
        :param ignore_invalid: Ignores invalid lines (empty lines)
        :param num: Parse up to num rows.
        :return: An iterator with the rows.
        """
        actions = []
        if standardize:
            actions.append(self._standardizer.apply_filter)
        if randomize:
            actions.append(conversions.randomize_smiles)

        with open(file_path, "r") as csv_file:
            for i, row in enumerate(csv_file):
                if i == num:
                    break
                splitted_row = row.rstrip().replace(",", " ").replace("\t", " ").split()
                smiles = splitted_row[0]
                for action in actions:
                    if smiles:
                        smiles = action(smiles)
                if smiles:
                    yield smiles
                elif not ignore_invalid:
                    yield None
