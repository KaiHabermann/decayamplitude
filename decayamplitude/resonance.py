from decayangle.decay_topology import Node
from decayamplitude.rotation import QN, Angular, clebsch_gordan
from typing import Union, Callable
from collections import namedtuple

LSTuple = namedtuple("LSTuple", ["l", "s"])

class Resonance:
    __instances = {}
    def __init__(self, node:Node, spin:Union[Angular, int] = None, parity:int = None, quantum_numbers:QN = None) -> None:
        self.node = node
        if quantum_numbers is None:
            if spin is None or parity is None:
                raise ValueError("Either quantum numbers or spin and parity must be provided")
            self.quantum_numbers = QN(spin, parity)
        else:
            self.quantum_numbers = quantum_numbers
        self.__daughter_qn = None
        self.id = Resonance.register(self)

    
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
    
    def helicity_from_ls(self, h0, h1, h2, couplings:dict[LSTuple, float], **arguments):

        q1, q2 = self.daughter_qn
        j1, j2 = q1.angular.angular_momentum, q2.angular.angular_momentum

        return sum(
            coupling * 
            self.lineshape(l,s,**arguments) * 
            clebsch_gordan(j1, h1, j2, h2, s, h1- h2) *
            clebsch_gordan(l, 0, s, h1 - h2, self.quantum_numbers.angular.angular_momentum, h1 - h2)
            for (l, s), coupling in couplings[self.tuple].items()
        )

    def construct_couplings(self, arguments:dict) -> dict[LSTuple, float]:
        """
        TEMPORARY VERSION FINAL SOLUTION NOT YET CLEAR

        We constuct the couplings for the resonance from real numbers found in the arguments dict
        Thus we need to know, which arguments belong to the couplings of the resonance
        The arguments are a dict of the form {id: {parameter_name: value}} where the id is the id of the resonance
        The ls couplings are a dict of the form {(l,s): value_r, } under the name ls_couplings
        """
        couplings = arguments[self.id]["ls_couplings"]


    def amplitude(self, h0, h1, h2, **arguments):
        couplings = self.construct_couplings(arguments)
        return self.helicity_from_ls(h0, h1, h2, couplings ,**arguments)
    
    @lineshape.setter
    def lineshape(self, lineshape_function:Callable):
        self.__lineshape = lineshape_function