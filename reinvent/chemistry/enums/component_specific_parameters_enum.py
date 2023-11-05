

class ComponentSpecificParametersEnum:
    __LOW = "low"
    __HIGH = "high"
    __K = "k"
    __TRANSFORMATION = "transformation"
    __SCIKIT = "scikit"
    __CLAB_INPUT_FILE = "clab_input_file"
    __COEF_DIV = "coef_div"
    __COEF_SI = "coef_si"
    __COEF_SE = "coef_se"
    __TRANSFORMATION_TYPE = "transformation_type"
    __DESCRIPTOR_TYPE = "descriptor_type"
    __TRUNCATE_LEFT = "truncate_left"
    __TRUNCATE_RIGHT = "truncate_right"
    __INTERPOLATION_MAP = "interpolation_map"

    # structural components
    # ---------
    __AZDOCK_CONFPATH = "configuration_path"
    __AZDOCK_DOCKERSCRIPTPATH = "docker_script_path"
    __AZDOCK_ENVPATH = "environment_path"
    __AZDOCK_DEBUG = "debug"

    @property
    def AZDOCK_DEBUG(self):
        return self.__AZDOCK_DEBUG

    @AZDOCK_DEBUG.setter
    def AZDOCK_DEBUG(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def AZDOCK_CONFPATH(self):
        return self.__AZDOCK_CONFPATH

    @AZDOCK_CONFPATH.setter
    def AZDOCK_CONFPATH(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def AZDOCK_DOCKERSCRIPTPATH(self):
        return self.__AZDOCK_DOCKERSCRIPTPATH

    @AZDOCK_DOCKERSCRIPTPATH.setter
    def AZDOCK_DOCKERSCRIPTPATH(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def AZDOCK_ENVPATH(self):
        return self.__AZDOCK_ENVPATH

    @AZDOCK_ENVPATH.setter
    def AZDOCK_ENVPATH(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def LOW(self):
        return self.__LOW

    @LOW.setter
    def LOW(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def HIGH(self):
        return self.__HIGH

    @HIGH.setter
    def HIGH(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def K(self):
        return self.__K

    @K.setter
    def K(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def TRANSFORMATION(self):
        return self.__TRANSFORMATION

    @TRANSFORMATION.setter
    def TRANSFORMATION(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def SCIKIT(self):
        return self.__SCIKIT

    @SCIKIT.setter
    def SCIKIT(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def CLAB_INPUT_FILE(self):
        return self.__CLAB_INPUT_FILE

    @CLAB_INPUT_FILE.setter
    def CLAB_INPUT_FILE(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def COEF_DIV(self):
        return self.__COEF_DIV

    @COEF_DIV.setter
    def COEF_DIV(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def COEF_SI(self):
        return self.__COEF_SI

    @COEF_SI.setter
    def COEF_SI(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def COEF_SE(self):
        return self.__COEF_SE

    @COEF_SE.setter
    def COEF_SE(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def TRANSFORMATION_TYPE(self):
        return self.__TRANSFORMATION_TYPE

    @TRANSFORMATION_TYPE.setter
    def TRANSFORMATION_TYPE(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def DESCRIPTOR_TYPE(self):
        return self.__DESCRIPTOR_TYPE

    @DESCRIPTOR_TYPE.setter
    def DESCRIPTOR_TYPE(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def TRUNCATE_LEFT(self):
        return self.__TRUNCATE_LEFT

    @TRUNCATE_LEFT.setter
    def TRUNCATE_LEFT(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def TRUNCATE_RIGHT(self):
        return self.__TRUNCATE_RIGHT

    @TRUNCATE_RIGHT.setter
    def TRUNCATE_RIGHT(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")

    @property
    def INTERPOLATION_MAP(self):
        return self.__INTERPOLATION_MAP

    @INTERPOLATION_MAP.setter
    def INTERPOLATION_MAP(self, value):
        raise ValueError("Do not assign value to a ComponentSpecificParametersEnum field")
