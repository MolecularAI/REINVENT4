class FilterTypesEnum:
    DEFAULT = "default"
    NEUTRALIZE_CHARGES = "neutralise_charges"
    GET_LARGEST_FRAGMENT = "get_largest_fragment"
    REMOVE_HYDROGENS = "remove_hydrogens"
    REMOVE_SALTS = "remove_salts"
    GENERAL_CLEANUP = "general_cleanup"
    UNWANTED_PATTERNS = "unwanted_patterns"
    VOCABULARY_FILTER = "vocabulary_filter"
    VALID_SIZE = "valid_size"
    HEAVY_ATOM_FILTER = "heavy_atom_filter"
    ALLOWED_ELEMENTS = "allowed_elements"

    # try to find the internal value and return
    def __getattr__(self, name):
        if name in self:
            return name

        raise AttributeError

    # prohibit any attempt to set any values
    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")
