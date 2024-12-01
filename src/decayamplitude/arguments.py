

class AmplitudeParameters:
    def __init__(self, function_arguments:list[str], couplings:dict[tuple[int], str]):
        self.__function_arguments = function_arguments
        self.__couplings = couplings
    
    def decode(arguments=None, couplings=None):
        if arguments is not None and couplings is not None:
            raise ValueError("")