Contributed scoring components
==============================

Description
-----------

These are various scoring components for REINVENT4 which should be mainly
considered as experimental and for demonstration purposes.

* UnwantedSubstructures: a thin wrapper around the various filter catalogs in RDKit.  What catalogs are available depend on the specific RDKit release.  The only parameter is the [catalogs](https://www.rdkit.org/docs/source/rdkit.Chem.rdfiltercatalog.html#rdkit.Chem.rdfiltercatalog.FilterCatalogParams.FilterCatalogs) one can choose from.

* NIBRSubstructureFilters: simple demo to show how to write one's own RDKit filter catalog.  The specific example was taken from [Contrib](https://github.com/rdkit/rdkit/tree/master/Contrib/NIBRSubstructureFilters) in the RDKit source.  The only parameter is a cutoff which serves as a delineation between "good" and "bad" molecules.  This is a rather simplistic approach and various improvements are possible.

* LillyMedchemRules: an interface to the [Lilly Medchem rules](https://github.com/IanAWatson/Lilly-Medchem-Rules) authored by Iain Watson while at Ely Lilly.  This is implementated as a scoring so will need a transform like the inverse sigmoid.  The only parameter is whether the relaxed rules should be used.  Requires the LillyMedchemRules Ruby script to be in the PATH.


Requirements
------------

To use this in REINVENT4 add the plugin path to PYTHONPATH e.g.
    
```shell
export PYTHONPATH=/location/to/REINVENT4/contrib
```
