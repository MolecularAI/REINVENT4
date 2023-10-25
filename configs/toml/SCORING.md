# Scoring Components in REINVENT4

This is a list of currently supported scoring components together with their
parameters.

* Qed: QED drug-likeness score (RDKit)
* SlogP: Crippen SLogP (RDKit)
* MolecularWeight: molecular weight (RDKit)
* TPSA: topological polar surface area (RDKit)
* GraphLength: topological distance matrix (RDKit)
* NumAtomStereoCenters: number of stereo centers (RDKit)
* HBondAcceptors: number of hydrogen bond acceptors (RDKit)
* HBondDonors: number of hydrogen bond donors (RDKit)
* NumRotBond: number of rotatable bonds (RDKit)
* Csp3: fraction of sp3 carbons (RDKit)
* numsp: number of sp hybridized atoms (RDKit)
* numsp2: number of sp2 hybridized atoms (RDKit)
* numsp3: number of sp3 hybridized atoms (RDKit)
* NumHeavyAtoms: number of heavy atoms (RDKit)
* NumHeteroAtoms: number of hetero atoms (RDKit)
* NumRings: number of total rings (RDKit)
* NumAromaticRings: number of aromatic rings (RDKit)
* NumAliphaticRings: number of aliphatic rings (RDKit)
* GroupCount: count how many times the SMARTS pattern is found
  * _smarts_: SMARTS pattern 
* MatchingSubstructure: penalty applied to final score when SMARTS pattern is found
  * _smarts_: list of SMARTS patterns
  * _use_chirality_: check for chirality
* PMI: principal moment of inertia to assess dimensionality (RDKit)
  * _property_: "npr1" or "npr2" to choose index 
* DockStream: generic docking interface for AutoDock Vina, rDock,
  OpenEye's Hybrid, Schrodinger's Glide and CCDC's GOLD
  * _configuration_path_: path for the Dockstream config json
  * _docker_script_path_: location of Dockstream "AZdock/docker.py" file
  * _docker_python_path_: python interpreter with Dockstream install, e.g. conda/envs/envname/bin/python
* TanimotoDistance: Tanimoto distance using the Morgan fingerprint (RDKit)
  * _smiles_: list of SMILES to match against
  * _radius_: Morgan fingerprint radius
  * _use_counts_: Morgan fingerprint, whether to use counts
  * _use_features_: Morgan fingerprint, whether to use featurs
* ChemProp: ChemProp D-MPNN models
  * _checkpoint_dir_: checkpoint directory with the models
  * _rdkit_2d_normalized_: whether to use RDKit 2D normalization
* CustomAlerts: SMARTS substructure filter applied to the total score
  * _smarts_: list of SMARTS
* ExternalProcess: generic component to run an external process for scoring
  * _executable_: name of the executable to run
  * _args_: command line arguments for the executable
* REST: generic REST interface (contributed by Syngenta)
  * _server_url_: URL
  * _server_port_: ports
  * _server_endpoint_: endpoint
  * _predictor_id_: request paramter
  * _predictor_version_: request paramter
  * _header_: request header
* Icolos: generic interface to Icolos
  * _name_: label of the score to extract
  * _executable_: Icolos executable
  * _config_file_: JSON config file for Icolos
* MMP: matched molecular pairs
  * _reference\_smiles_:
  * _num\_of\_cuts_:
  * _max\_variable\_heavies_:
  * _max\_variable\_ratio_:
* Qptuna: QSAR models with Qptuna
  * _mode\_file_: model file name
* ReactionFilter: reaction filter for Libinvent, applied to total score
  * _tyoe_: filter type
  * _reaction\_smarts_: RDKit reaction SMARTS
* ROCSSimilarity: OpenEye ROCS
  * _color\_weight_: float between 0-1, default 0.5, weighting between shape and color scores
  * _shape\_weight_: float between 0-1, default 0.5, weighting between shape and color scores
  * _custom\_cff_: path to custom ROCs forecfield, optional
  * _max\_stereocenters_: max number of stereo centers to enumerate
  * _ewindow_: energy window for conformers (kJ/mol)
  * _maxconfs_: max number of confs per compound
  * _rocs\_input_: input file, sdf or sq 
  * _similarity\_measure_: how to compare shapes. Must be Tanimoto, RefTversky or FitTversky  
* SAScore:  Ertl's synthesizability score (higher is more difficult). based on https://doi.org/10.1186/1758-2946-1-8.

Linkinvent specific physchem properties:
* FragmentMolecularWeight
* FragmentNumAliphaticRings
* FragmentGraphLength
* FragmentEffectiveLength
* FragmentLengthRatio
* FragmentHBondAcceptors
* FragmentHBondDonors
* FragmentNumRotBond
* Fragmentnumsp
* Fragmentnumsp2
* Fragmentnumsp3
* FragmentNumRings
* FragmentNumAromaticRings
* FragmentNumAliphaticRings

## Transformation functions

* _sigmoid_, _reverse\_sigmoid_, _double\_sigmoid_: one- and two-sided sigmoid functions
* _left\_step_, _right\_step_, _step_: one- and two-sided step functions
* _value\_mapping_: map labels/categories to numbers, number must be be in the range \[0.0, 1.0\]


##  Aggregation functions

* _arithmetic\_mean_: weighted arithemtic mean
* _geometric\_mean_: weighted geometric mean
