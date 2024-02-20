"""Input file reader for scoring

Assumes two formats: SMI or CSV
SMI file is assumed to have either one or two columns, SMILES in the first, no header!
CSV file is always assumed to have header

FIXME: review again if both formats should be treated as one singular, tabular
       format
"""

__all__ = ["TabFileReader", "get_dialect", "write_csv"]
import csv
from typing import List, Callable

ACTION_CALLABLES = List[Callable[[str], str]]


class TabFileReader:
    """Tabular file reader

    SMILES: one or two column format, SMILES column strictly in first column
            no header!
    CSV: expects header, SMILES column name is optional
    """

    def __init__(
        self,
        filename: str,
        actions: ACTION_CALLABLES = None,
        header: bool = False,
        delimiter: str = "\t",
        smiles_column: str = "SMILES",
    ):
        """
        :param filename: name of SMILES file
        :param actions: list of callables to apply to each SMILES
        :param header: presence of header
        :param delimiter: column delimiter in SMILES file
        param smiles_columns: name of column containing the SMILES
        """

        self.filename = filename
        self.actions = actions
        self.header = header
        self.delimiter = delimiter
        self.smiles_column = smiles_column

        self._rows = []
        self._smilies = []
        self._header_line = None

    def read(self):
        with open(self.filename, "r") as tabfile:
            if self.header:
                dialect = get_dialect(tabfile)
                reader = csv.reader(tabfile, dialect=dialect)
                header = next(reader)
                self._header_line = header

                try:
                    col_num = header.index(self.smiles_column)
                except ValueError:
                    raise RuntimeError(f"{__name__}: SMILES column could not be found")
            else:
                col_num = 0
                reader = csv.reader(tabfile, delimiter=self.delimiter)

            for row in reader:
                smiles = row[col_num].strip()

                if self.actions:
                    for action in self.actions:
                        if callable(action) and smiles:
                            smiles = action(smiles)

                self._smilies.append(smiles)

                if self.header:
                    self._rows.append(row)
                else:  # currently synonymous with SMILES file format
                    self._rows.append([row[0], " ".join(row[1:])])

    @property
    def rows(self) -> List:
        return self._rows

    @property
    def smilies(self) -> List[str]:
        return self._smilies

    @property
    def header_line(self) -> str | None:
        if not self._header_line:  # SMILES file format
            self._header_line = ["SMILES", "Comment"]

        return self._header_line


def get_dialect(csvfile) -> csv.Dialect:
    """Guess the CSV dialect from a file handle

    NOTE: for this to work the sniffer must be fed with only the header line

    :param csvfile: file handle
    :returns: the CSV dialect
    """

    sniffer = csv.Sniffer()

    line = csvfile.read(4096)
    pos = line.find("\n")

    if pos < 0:
        raise RuntimeError(f"{__name__}: incomplete header read")

    dialect = sniffer.sniff(line[:pos])

    csvfile.seek(0)

    return dialect


def write_csv(filename, header, rows):
    with open(filename, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)
