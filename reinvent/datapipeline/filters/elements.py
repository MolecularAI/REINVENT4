"""Element handling"""

from typing import Sequence

from rdkit import Chem


BASE_ELEMENTS = {"C", "O", "N", "S", "F", "Cl", "Br", "I"}
_PT = Chem.GetPeriodicTable()
PERIODIC_TABLE = {
    elem: _PT.GetAtomicWeight(elem) for elem in [_PT.GetElementSymbol(an) for an in range(1, 119)]
}


def valid_elements(elements: Sequence[str]) -> bool:
    """Check if all elements are valid as per periodic table

    :param elements: sequence of elements
    :returns: true if all elements are valid, false otherwise
    """

    for element in elements:
        if element not in PERIODIC_TABLE.keys():
            return False

    return True
