import pathlib
import numpy as np
from openeye import oechem, oeomega, oeshape, oequacpac
from typing import Optional, List


class ROCSOverlay:
    """
    This class contains all methods needed for running OpenEye ROCS
    based on the OE python API. Env variable "OE_LICENSE" must be set
    before running this code.

    :param rocs_input: path to either sdf or sq input file
    :param color_weight: weighting for color score
    :param shape_weight: weighting for color score
    :param max_stereocenters: max # stereocenters to enum.
    :param ewindow: energy window around min for conf search, kJ/mol
    :param maxconfs: max # of conformers to search
    :param similarity_measure: Tanimoto, RefTversky or FitTversky. see ROCs documentation
    :params custom_cff: path to custom ROCs color force field, if desired.
    """

    def __init__(
        self,
        rocs_input: str,
        color_weight: float = 0.5,
        shape_weight: float = 0.5,
        max_stereocenters: int = 10,
        ewindow: float = 10.0,
        maxconfs: int = 200,
        similarity_measure: str = "Tanimoto",
        custom_cff: Optional[str] = None,
    ):
        self.rocs_input = rocs_input
        self.shape_weight = shape_weight
        self.color_weight = color_weight
        self.similarity_measure = similarity_measure
        self.max_stereocenters = max_stereocenters
        self.ewindow = ewindow
        self.maxconfs = maxconfs
        self.custom_cff = custom_cff

        # run all OE initialization routines
        self._set_ff()
        self._setup_omega()
        self._prepare_overlay()
        self._setup_similarity()
        oechem.OEThrow.SetLevel(10000)

    def _set_ff(self):
        # initialize the OE omega objects,
        # load custom cff if required

        overlay_prep = oeshape.OEOverlapPrep()
        if self.custom_cff is None:
            cff_path = oeshape.OEColorFFType_ImplicitMillsDean
        else:
            cff_path = self.custom_cff
        cff = oeshape.OEColorForceField()
        if cff.Init(cff_path):
            overlay_prep.SetColorForceField(cff)
        else:
            raise ValueError("Custom color force field initialisation failed")
        self.overlay_prep = overlay_prep

    def get_omega_confs(self, imol):
        enantiomers = list(oeomega.OEFlipper(imol.GetActive(), self.max_stereocenters, False, True))
        for k, enantiomer in enumerate(enantiomers):
            # Any other simpler way to combine and add all conformers to imol have failed !!
            # Failure = Creates conformers with wrong indices and wrong connections
            enantiomer = oechem.OEMol(enantiomer)
            ret_code = self.omega.Build(enantiomer)
            if ret_code == oeomega.OEOmegaReturnCode_Success:
                if k == 0:
                    imol = oechem.OEMol(enantiomer.SCMol())
                    imol.DeleteConfs()
                for x in enantiomer.GetConfs():
                    imol.NewConf(x)
        self.omega.Build(imol)
        return imol

    def _setup_similarity(self):
        if self.similarity_measure == "Tanimoto":
            shape_getter = "GetTanimoto"
            color_getter = "GetColorTanimoto"
            combo_getter = "OEHighestTanimotoCombo"
        elif self.similarity_measure == "RefTversky":
            shape_getter = "GetRefTversky"
            color_getter = "GetRefColorTversky"
            combo_getter = "OEHighestRefTverskyCombo"
        elif self.similarity_measure == "FitTversky":
            shape_getter = "GetFitTversky"
            color_getter = "GetFitColorTversky"
            combo_getter = "OEHighestFitTverskyCombo"
        else:
            raise ValueError("similarity measure must be Tanimoto, RefTversky or FitTversky")

        self.shape_getter = shape_getter
        self.color_getter = color_getter
        self.combo_getter = combo_getter

    def _prep_sdf_file(self, outmol, score, smile, best_score_shape, best_score_color):
        score.Transform(outmol)
        oechem.OESetSDData(outmol, "Smiles", smile)
        oechem.OESetSDData(outmol, "Shape", "%-.3f" % best_score_shape)
        oechem.OESetSDData(outmol, "Color", "%-.3f" % best_score_color)

    def _setup_omega(self):
        omegaOpts = oeomega.OEOmegaOptions()
        omegaOpts.SetStrictStereo(False)
        omegaOpts.SetEnergyWindow(self.ewindow)
        omegaOpts.SetMaxConfs(self.maxconfs)
        omegaOpts.GetTorDriveOptions().SetUseGPU(False)
        self.omega = oeomega.OEOmega(omegaOpts)

    def _setup_overlay_from_shape_query(self):
        qry = oeshape.OEShapeQuery()
        overlay = oeshape.OEOverlay()
        if oeshape.OEReadShapeQuery(self.rocs_input, qry):
            overlay.SetupRef(qry)
        else:
            raise ValueError("error reading in SQ file")
        self.rocs_overlay = overlay

    def _setup_overlay_from_sdf_file(self):
        input_stream = oechem.oemolistream()
        input_stream.SetFormat(oechem.OEFormat_SDF)
        input_stream.SetConfTest(oechem.OEAbsoluteConfTest(compTitles=False))
        refmol = oechem.OEMol()
        if input_stream.open(self.rocs_input):
            oechem.OEReadMolecule(input_stream, refmol)
        else:
            raise ValueError("error reading in ROCS sdf file")
        self.overlay_prep.Prep(refmol)
        overlay = oeshape.OEMultiRefOverlay()
        overlay.SetupRef(refmol)
        self.rocs_overlay = overlay

    def _prepare_overlay(self):
        file_extension = pathlib.Path(self.rocs_input).suffix.lower()
        if file_extension == ".sdf":
            self._setup_overlay_from_sdf_file()
        elif file_extension == ".sq":
            self._setup_overlay_from_shape_query()

    def calculate_rocs_score(self, smiles: List[str]) -> np.array:
        scores = []
        outmols = []
        for smile in smiles:
            imol = oechem.OEMol()
            best_score = 0.0
            if oechem.OESmilesToMol(imol, smile):
                oequacpac.OEGetReasonableProtomer(imol)
                imol = self.get_omega_confs(imol)
                self.overlay_prep.Prep(imol)

                score = oeshape.OEBestOverlayScore()
                self.rocs_overlay.BestOverlay(score, imol, getattr(oeshape, self.combo_getter)())
                best_score_shape = min(getattr(score, self.shape_getter)(), 1)
                best_score_color = min(getattr(score, self.color_getter)(), 1)
                best_score = (
                    (self.shape_weight * best_score_shape) + (self.color_weight * best_score_color)
                ) / (self.shape_weight + self.color_weight)

                outmol = oechem.OEGraphMol(imol.GetConf(oechem.OEHasConfIdx(score.GetFitConfIdx())))
                self._prep_sdf_file(
                    outmol,
                    score=score,
                    smile=smile,
                    best_score_shape=best_score_shape,
                    best_score_color=best_score_color,
                )
                outmols.append(outmol)
                scores.append(best_score)

        return np.array(scores)
