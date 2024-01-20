__all__ = ["MmpPairGenerator"]
from argparse import Namespace, RawDescriptionHelpFormatter, ArgumentParser
from time import time
import logging
import tempfile

from mmpdblib.do_fragment import fragment_command
from mmpdblib.do_index import index_command
import numpy as np
import os
import pandas as pd

from .pair_generator import PairGenerator


logger = logging.getLogger(__name__)


class MmpPairGenerator(PairGenerator):
    """Generator of molecule pairs according to MMP similarity"""

    def __init__(
        self,
        hac: int,
        ratio: float,
        max_radius: int = 5,
        add_same: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """__init__.

        :param hac: number of max heavy atoms to consider while generating the MMP
        :type hac: int
        :param ratio: keeps all the pairs such that mmp ratio is <= ratio
        :type ratio: float
        :param max_radius:
        :type max_radius: int
        :param add_same: whether to inlcude the pairs (s,s) or not
        :type add_same: bool
        :rtype: None
        """
        super().__init__(*args, **kwargs)
        if (ratio >= 1.0) or (ratio <= 0):
            raise ValueError("`ratio` must be in (0,1)")
        self.hac = hac
        self.ratio = ratio
        self.max_radius = max_radius
        self.add_same = add_same
        self.tmp_folder = tempfile.TemporaryDirectory(prefix="mmp_tmp", dir="")

    def build_pairs(self, smiles: list, *, processes: int = 8) -> pd.DataFrame:
        """Build pairs starting from a column containing
        N smiles. The maximum number of pairs is therfore
        N(N-1)/2.

        :param smiles: a list containing smiles
        :type smiles: list
        :param processes: number of process for parallelizing the construction of pairs
        :type processes: int
        :rtype: pd.DataFrame
        """
        if len(smiles) == 0:
            raise ValueError("The smiles list is empty")

        logger.info(f"Creating MMP pairs with {processes:d} processes...")

        mmp_input_file = self._prepare_mmp_input(smiles)
        mmp_fragments_file = self._mmp_fragments(mmp_input_file, num_jobs=processes)
        mmp_index_file = self._mmp_index(mmp_fragments_file)

        df = self._remove_duplicates(mmp_index_file)

        if self.add_same:
            frontier = set()
            new_table = []
            for i in range(len(df)):
                source = df.loc[i]["Source_Mol"]
                target = df.loc[i]["Target_Mol"]
                if not source in frontier:
                    df_copy = df.loc[i].copy()
                    df_copy["Target_Mol"] = source
                    df_copy["Target_Mol_ID"] = df_copy["Source_Mol_ID"]
                    new_table.append(df_copy.values)
                    frontier.add(source)
                if not target in frontier:
                    df_copy = df.loc[i].copy()
                    df_copy["Source_Mol"] = target
                    df_copy["Source_Mol_ID"] = df_copy["Target_Mol_ID"]
                    new_table.append(df_copy.values)
                    frontier.add(target)
            df_same = pd.DataFrame(new_table, columns=df.columns)
            df = pd.concat((df, df_same)).reset_index(drop=True)
        df = self.filter(df)
        self.tmp_folder.cleanup()

        logger.info("MMP pairs created")
        return df

    def _prepare_mmp_input(self, smiles):
        df_output = pd.DataFrame(
            np.hstack((np.array(smiles)[:, None], np.arange(len(smiles))[:, None])),
            columns=["smi", "id"],
        )
        out_file = f"{self.tmp_folder.name}/mmp_input_{time():.5f}.smi"
        while os.path.exists(out_file):
            out_file = f"{self.tmp_folder.name}/mmp_input_{time():.5f}.smi"
        df_output.to_csv(out_file, index=False, header=False)
        return out_file

    def _mmp_fragments(self, input_file, *, nofcuts=1, num_jobs=-1):
        if num_jobs == -1:
            num_jobs = os.cpu_count()

        output_file = os.path.splitext(input_file)[0] + ".fragments"

        parser = ArgumentParser(
            prog="mmpdb fragment",
            usage=None,
            description=None,
            formatter_class=RawDescriptionHelpFormatter,
            conflict_handler="error",
            add_help=True,
        )

        args = Namespace(
            cache=None,
            command=fragment_command,
            cut_rgroup=None,
            cut_rgroup_file=None,
            cut_smarts=None,
            delimiter="comma",
            format=None,
            has_header=False,
            max_heavies=None,
            max_rotatable_bonds=None,
            min_heavies_per_const_frag=None,
            num_cuts=nofcuts,
            num_jobs=num_jobs,
            out=None,
            quiet=False,
            rotatable_smarts=None,
            salt_remover=None,
            structure_filename=input_file,
            output=output_file,
            subparser=parser,
        )
        fragment_command(parser, args)
        if os.path.exists(output_file):
            return output_file
        else:
            return None

    def _mmp_index(self, input_file):
        hac = self.hac
        ratio = self.ratio
        max_radius = self.max_radius
        output_file = os.path.splitext(input_file)[0] + ".index.csv"
        parser = ArgumentParser(
            prog="mmpdb index",
            usage=None,
            description=None,
            formatter_class=RawDescriptionHelpFormatter,
            conflict_handler="error",
            add_help=True,
        )

        args = Namespace(
            command=index_command,
            max_frac_trans=None,
            max_heavies_transf=None,
            max_radius=max_radius,
            max_variable_heavies=hac,
            max_variable_ratio=ratio,
            memory=False,
            min_variable_heavies=None,
            min_variable_ratio=None,
            out="csv",
            properties=None,
            quiet=False,
            smallest_transformation_only=True,
            subparser=parser,
            fragment_filename=input_file,
            output=output_file,
            symmetric=True,
            title=None,
        )
        index_command(parser, args)
        if os.path.exists(output_file):
            return output_file
        else:
            return None

    def _remove_duplicates(self, input_path):
        try:
            results = pd.read_csv(input_path, sep="\t", header=None)
        except EmptyDataError:
            results = pd.DataFrame()
        if len(results) > 0:
            results.columns = [
                "Source_Mol",
                "Target_Mol",
                "Source_Mol_ID",
                "Target_Mol_ID",
                "Transformation",
                "Core",
            ]
            # Remove duplicated  pairs
            results["Source_R_len"] = results["Transformation"].apply(len)
            results = results.sort_values("Source_R_len")
            results = results.drop_duplicates(subset=["Source_Mol_ID", "Target_Mol_ID"])
            results = results.reset_index(drop=True)
        return results

    def get_params(self):
        params = {
            "hac": self.hca,
            "ratio": self.ratio,
            "max_radius": self.max_radius,
            "add_same": self.add_same,
        }
        return {*params, *super().get_params()}
