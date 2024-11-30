from decayangle.decay_topology import Node
from decayamplitude.rotation import QN, Angular, clebsch_gordan
from typing import Union, Callable
from collections import namedtuple

LSTuple = namedtuple("LSTuple", ["l", "s"])

class Resonance:
    __instances = {}
    __parameter_names = {}
    def __init__(self, node:Node, spin:Union[Angular, int] = None, parity:int = None, quantum_numbers:QN = None, lineshape = None, argnames = None) -> None:
        self.node = node
        if quantum_numbers is None:
            if spin is None or parity is None:
                raise ValueError("Either quantum numbers or spin and parity must be provided")
            self.quantum_numbers = QN(spin, parity)
        else:
            self.quantum_numbers = quantum_numbers
        self.__daughter_qn = None
        self.__id = Resonance.register(self)
        self.__lineshape = None
        if lineshape is not None:
            if argnames is None:
                raise ValueError("If a lineshape is provided, the argument names must be provided as well")
            self.register_lineshape(lineshape, argnames)
                
    def argument_list(self, arguments:dict) -> list:
        return [arguments[name] for name in self.__parameter_names]
    
    @property
    def id(self) -> int:
        return self.__id
    
    @id.setter
    def id(self, value:int):
        raise ValueError("The id of a resonance cannot be changed")
    
    id.deleter
    def id(self):
        raise ValueError("The id of a resonance cannot be deleted")

    @classmethod
    def register(cls, obj) -> int:
        instance_id = len(cls.__instances)
        cls.__instances[instance_id] = obj
        return instance_id


    @property
    def lineshape(self) -> Callable:
        return self.__lineshape
    
    @property
    def daughter_qn(self) -> list[QN]:
        if self.__daughter_qn is None:
            raise ValueError("Daughter quantum numbers not set! This should happen in the DecayChainNode class. The resonance class should only be used as part of a DecayChain!")
        return self.__daughter_qn
    
    @daughter_qn.setter
    def daughter_qn(self, daughters: list[QN]):
        self.__daughter_qn = daughters
    
    def helicity_from_ls(self, h0, h1, h2, couplings:dict[LSTuple, float], arguments:dict):

        q1, q2 = self.daughter_qn
        j1, j2 = q1.angular.angular_momentum, q2.angular.angular_momentum

        return sum(
            coupling * 
            self.lineshape(l,s,*self.argument_list(arguments)) * 
            (l + 1) ** 0.5 /
            (self.spin + 1) ** 0.5 *
            clebsch_gordan(j1, h1, j2, h2, s, h1- h2) *
            clebsch_gordan(l, 0, s, h1 - h2, self.quantum_numbers.angular.angular_momentum, h1 - h2)
            for (l, s), coupling in couplings[self.tuple].items()
        ) (-1) ** ((j2 - h2) / 2)

    def construct_couplings(self, arguments:dict) -> dict[LSTuple, float]:
        """
        TEMPORARY VERSION FINAL SOLUTION NOT YET CLEAR

        We constuct the couplings for the resonance from real numbers found in the arguments dict
        Thus we need to know, which arguments belong to the couplings of the resonance
        The arguments are a dict of the form {id: {parameter_name: value}} where the id is the id of the resonance
        The ls couplings are a dict of the form {(l,s): value_r } under the name ls_couplings
        """
        couplings = arguments[self.id]["ls_couplings"]
        return {LSTuple(*key): value for key, value in couplings.items()}


    def amplitude(self, h0, h1, h2, arguments:dict):
        couplings = self.construct_couplings(arguments)
        return self.helicity_from_ls(h0, h1, h2, couplings ,arguments)
    
    def register_lineshape(self, lineshape_function:Callable, parameter_names: list[str]):
        if self.__lineshape is not None:
            raise ValueError("Lineshape already set")
        self.__lineshape = lineshape_function
        self.__parameter_names = {}
        for parameter_name in parameter_names:
            if parameter_name not in self.__parameter_names:
                type(self).__parameter_names[parameter_name] = self
        return self