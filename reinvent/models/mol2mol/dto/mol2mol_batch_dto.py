from dataclasses import dataclass, astuple

import torch


@dataclass
class Mol2MolBatchDTO:
    input: torch.Tensor
    input_mask: torch.Tensor
    output: torch.Tensor
    output_mask: torch.Tensor
    tanimoto: torch.Tensor = None

    def __iter__(self):
        return iter(astuple(self))

    def __len__(self):
        return len(self.input)
