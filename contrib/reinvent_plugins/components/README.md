Contributed scoring components
==============================

Description
-----------

These are various scoring components for REINVENT4 which should be mainly
considered as experimental and for demonstration purposes.

* UnwantedSubstructures: a thin wrapper around the various filter catalogs in RDKit.  What
catalogs are available depend on the specific RDKit release.  The only parameter is the
[catalogs](https://www.rdkit.org/docs/source/rdkit.Chem.rdfiltercatalog.html#rdkit.Chem.rdfiltercatalog.FilterCatalogParams.FilterCatalogs)
one can choose from.

* NIBRSubstructureFilters: simple demo to show how to write one's own RDKit filter catalog.
The specific example was taken from
* [Contrib](https://github.com/rdkit/rdkit/tree/master/Contrib/NIBRSubstructureFilters) in
the RDKit source.  The only parameter is a cutoff which serves as a delineation between
"good" and "bad" molecules.  This is a rather simplistic approach and various improvements
are possible.

* LillyMedchemRules: an interface to the [Lilly Medchem rules](https://github.com/IanAWatson/Lilly-Medchem-Rules) authored by Ian Watson while at Ely Lilly.  This is implementated as a scoring so will need a transform like the inverse sigmoid.  The only parameter is whether the relaxed rules should be used.  Needs to have the `LILLY_MEDCHEM_RULES_ROOT` environment variable set to the top directory of the Lilly-Medchem-Rules installation.

* LillyDescriptors: 259 descriptors from Ely Lilly's library of Chemoinformatics [LillyMol](https://github.com/EliLillyCo/LillyMol).  The only parameter is the wanted descriptors (one per endpoint).  The `LILLY_MOL_ROOT` needs to be set to the top directory of the LillyMol installation.  Unfortunately, there is little documentation available.  Descriptor groups:
    * adjacent to ring fusion descriptors
    * bonds between rings descriptors
    * all formal charge descriptors
    * all expensive chirality perception descriptors
    * planar fused ring descriptors
    * atomic crowding descriptors
    * donor acceptor derived descriptors
    * donor/acceptor derived descriptors
    * simplistic donor/acceptor derived descriptors
    * ncon and fncon descriptors
    * polar bond derived descriptors
    * Novartis polar surface area descriptors
    * partial symmetry derived descriptors
    * Ramey (element count) descriptors
    * ring chain join descriptors
    * ring fusion descriptors
    * ring substitution descriptors
    * ring substitution ratio descriptors
    * spinach related descriptors
    * specific group descriptors
    * symmetry related descriptors
    * xlogp

* LillyPAINS: uses tsubstructure from LillyMol to match input SMILES with PAINS patterns.
Multiple matches are not handled specially at the moment.  There are only 63
patterns available which is less than the RDKit PAINS filter has but the advantage is that
this component provides scores for 12 different assays and 3 enrichnents rather than just a
single yes/no score.  The only parameter is the assay name (one per endpoint).
    * Alpha
    * ELISA
    * FB
    * FP
    * FRET
    * SPA
    * Alpha_HS
    * ELISA_HS
    * FB_HS
    * FP_HS
    * FRET_HS
    * SPA_HS
    * OverallActivityEnrichment
    * QCEnrichment
    * HSEnrichment
    * TotalScore


Requirements
------------

To use this in REINVENT4 add the plugin path to PYTHONPATH e.g.
    
```shell
export PYTHONPATH=/location/to/REINVENT4/contrib
```
