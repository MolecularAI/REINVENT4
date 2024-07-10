from dataclasses import dataclass, astuple

import torch


@dataclass
class BatchDTO:
    input: torch.Tensor
    input_mask: torch.Tensor
    output: torch.Tensor
    output_mask: torch.Tensor
    tanimoto: torch.Tensor = None  # Mol2Mol rankingloss model

    def __iter__(self):
        return iter(astuple(self))

    def __len__(self):
        return len(self.input)
