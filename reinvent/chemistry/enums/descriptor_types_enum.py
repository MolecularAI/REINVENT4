class DescriptorTypesEnum:
    _ECFP = "ecfp"
    _ECFP_COUNTS = "ecfp_counts"
    _MACCS_KEYS = "maccs_keys"
    _AVALON = "avalon"

    @property
    def ECFP(self):
        return self._ECFP

    @ECFP.setter
    def ECFP(self, value):
        raise ValueError("Do not assign value to a DescriptorTypesEnum field")

    @property
    def ECFP_COUNTS(self):
        return self._ECFP_COUNTS

    @ECFP_COUNTS.setter
    def ECFP_COUNTS(self, value):
        raise ValueError("Do not assign value to a DescriptorTypesEnum field")

    @property
    def MACCS_KEYS(self):
        return self._MACCS_KEYS

    @MACCS_KEYS.setter
    def MACCS_KEYS(self, value):
        raise ValueError("Do not assign value to a DescriptorTypesEnum field")

    @property
    def AVALON(self):
        return self._AVALON

    @AVALON.setter
    def AVALON(self, value):
        raise ValueError("Do not assign value to a DescriptorTypesEnum field")