from typing import Dict, List, Optional
import logging

from rdkit.Chem.rdmolfiles import MolToSmiles

from reinvent.chemistry import conversions
from reinvent.chemistry.standardization.filter_configuration import FilterConfiguration
from reinvent.chemistry.standardization.filter_registry import FilterRegistry


logger = logging.getLogger(__name__)


class RDKitStandardizer:
    def __init__(
        self, filter_configs: Optional[List[FilterConfiguration]], isomeric=False, *args, **kwargs
    ):
        self._filter_configs = self._set_filter_configs(filter_configs)
        self._filters = self._load_filters(self._filter_configs)
        self.isomeric = isomeric

        for config in self._filter_configs:
            logger.info(f"Applying filter {config.name} to input SMILES")

        if self.isomeric:
            logger.info("Stereochemistry kept in input SMILES")

    def apply_filter(self, smile: str) -> str:
        molecule = conversions.smile_to_mol(smile)

        for config in self._filter_configs:
            if molecule:
                rdkit_filter = self._filters[config.name]

                if config.parameters:
                    molecule = rdkit_filter(molecule, **config.parameters)
                else:
                    molecule = rdkit_filter(molecule)
            else:
                message = f'"{config.name}" filter: {smile} is invalid'
                logger.warning(message)

        if not molecule:
            message = f'"{self._filter_configs[-1].name}" filter: {smile} is invalid'
            logger.warning(message)
            return None

        valid_smile = MolToSmiles(molecule, isomericSmiles=self.isomeric)

        return valid_smile

    def _set_filter_configs(self, filter_configs):
        return (
            filter_configs
            if filter_configs
            else [FilterConfiguration(name="default", parameters={})]
        )

    def _load_filters(self, filter_configs: List[FilterConfiguration]) -> Dict:
        registry = FilterRegistry()

        return {
            filter_config.name: registry.get_filter(filter_config.name)
            for filter_config in filter_configs
        }
