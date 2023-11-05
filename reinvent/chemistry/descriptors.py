from typing import List, Dict

import numpy as np
from rdkit import DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, MACCSkeys, Mol
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint

from reinvent.chemistry.enums.component_specific_parameters_enum import (
    ComponentSpecificParametersEnum,
)
from reinvent.chemistry.enums.descriptor_types_enum import DescriptorTypesEnum


class Descriptors:
    """Molecular descriptors.

    The descriptors in this class are mostly RDKit fingerprints used as inputs to
    scikit-learn predictive models. Since scikit-learn predictive models take
    numpy arrays as input, RDKit fingerprints are converted to numpy arrays.
    """

    def __init__(self):
        self._descriptor_types = DescriptorTypesEnum()
        self._specific_parameters = ComponentSpecificParametersEnum()

    def load_descriptor(self, parameters: {}):
        descriptor_type = parameters.get(
            self._specific_parameters.DESCRIPTOR_TYPE, self._descriptor_types.ECFP_COUNTS
        )
        registry = self._descriptor_registry()
        descriptor = registry[descriptor_type]
        return descriptor

    def _descriptor_registry(self) -> dict:
        descriptor_list = dict(
            ecfp=self.molecules_to_fingerprints,
            ecfp_counts=self.molecules_to_count_fingerprints,
            maccs_keys=self.maccs_keys,
            avalon=self.avalon,
        )
        return descriptor_list

    def maccs_keys(self, molecules: List[Mol], parameters: {}):
        fps = [MACCSkeys.GenMACCSKeys(mol) for mol in molecules]
        fingerprints = [self._numpy_fingerprint(fp, dtype=np.int32) for fp in fps]
        return fingerprints

    def avalon(self, molecules: List[Mol], parameters: {}):
        size = parameters.get("size", 512)
        fps = [pyAvalonTools.GetAvalonFP(mol, nBits=size) for mol in molecules]
        fingerprints = [self._numpy_fingerprint(fp, dtype=np.int32) for fp in fps]
        return fingerprints

    def molecules_to_fingerprints(self, molecules: List[Mol], parameters: {}) -> List[np.ndarray]:
        radius = parameters.get("radius", 3)
        size = parameters.get("size", 2048)
        fp_bits = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, size) for mol in molecules]
        fingerprints = [self._numpy_fingerprint(fp, dtype=np.int32) for fp in fp_bits]
        return fingerprints

    def molecules_to_count_fingerprints_ori(
        self, molecules: List[Mol], parameters: {}
    ) -> np.ndarray:
        """Morgan Count Fingerprints.

        This is "original" implementation, with own hashing code.
        """
        radius = parameters.get("radius", 3)
        useCounts = parameters.get("use_counts", True)
        useFeatures = parameters.get("use_features", True)
        size = parameters.get("size", 2048)
        fps = [
            AllChem.GetMorganFingerprint(mol, radius, useCounts=useCounts, useFeatures=useFeatures)
            for mol in molecules
        ]
        nfp = np.zeros((len(fps), size), np.int32)
        for i, fp in enumerate(fps):
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % size
                nfp[i, nidx] += int(v)
        return nfp

    def molecules_to_count_fingerprints(
        self, molecules: List[Mol], parameters: Dict
    ) -> List[np.ndarray]:
        """Morgan Count Fingerprints from RDKit.

        This implementation uses RDKit's hashing through GetHashedMorganFingerprint.
        See https://stackoverflow.com/a/55119975
        """

        radius = parameters.get("radius", 3)
        useFeatures = parameters.get("use_features", True)
        size = parameters.get("size", 2048)
        fps = [
            GetHashedMorganFingerprint(
                mol,
                radius,
                nBits=size,
                useFeatures=useFeatures,
            )
            for mol in molecules
        ]
        fingerprints = [self._numpy_fingerprint(fp, dtype=np.int32) for fp in fps]
        return fingerprints

    def _numpy_fingerprint(self, rdkit_fingerprint, dtype=None) -> np.ndarray:
        """Converts RDKit fingerprint to numpy array."""

        numpy_fp = np.zeros((0,), dtype=dtype)  # Initialize empty array, RDKit will resize it.
        DataStructs.ConvertToNumpyArray(rdkit_fingerprint, numpy_fp)
        return numpy_fp
