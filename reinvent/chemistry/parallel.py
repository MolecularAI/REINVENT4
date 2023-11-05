from rdkit import Chem

class ParallelMoleculeHandler:

    @staticmethod
    def prepare_local_parallel():
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
