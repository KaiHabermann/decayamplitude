from __future__ import annotations

from decayangle.decay_topology import Node, HelicityAngles
from decayamplitude.rotation import QN, Angular, clebsch_gordan, wigner_capital_d, convert_angular
from typing import Union, Callable, Literal
from collections import namedtuple
from functools import cached_property
from decayamplitude.utils import sanitize


LSTuple = namedtuple("LSTuple", ["l", "s"])
HelicityTuple = namedtuple("HelicityTuple", ["h1", "h2"])

class Resonance:
    __instances = {}
    __named_instances = {}
    __parameter_names = {}

    @classmethod
    def get_instance(cls, id:int) -> "Resonance":
        return cls.__instances[id]

    def __init__(self, node:Node, spin:Union[Angular, int] = None, parity:int = None, quantum_numbers:QN = None, lineshape = None, argnames = None, name = None, preserve_partity=True, scheme:Literal["ls", "helicity"]="ls") -> None:
        self.node = node
        self.preserve_partity = preserve_partity
        if quantum_numbers is None:
            if spin is None or parity is None:
                raise ValueError("Either quantum numbers or spin and parity must be provided")
            self.quantum_numbers = QN(spin, parity)
        else:
            self.quantum_numbers = quantum_numbers
        self.__daughters = None
        self.__lineshape = None
        if name is not None:
            self.__name = name
        else: 
            # we explicitly want to differentiate resonances with and without name
            self.__name = None
        self.__id = Resonance.register(self)

        if lineshape is not None:
            if argnames is None:
                raise ValueError("If a lineshape is provided, the argument names must be provided as well")
            self.register_lineshape(lineshape, argnames)
            self.__lineshape = lineshape
            self.__parameter_names = argnames
        if scheme not in ["ls", "helicity"]:
            raise ValueError("scheme must be either 'ls' or 'helicity'")
        self.__scheme = scheme

    @property
    def parameter_names(self) -> list[str]:
        return self.__parameter_names
                
    def argument_list(self, arguments:dict) -> list:
        return [arguments[name] for name in self.__parameter_names]

    @property
    def scheme(self) -> str:
        return self.__scheme

    @property
    def name(self) -> str:
        if self.__name is None:
            return f"ID_{self.id}"
        return self.__name
    
    @cached_property
    def sanitized_name(self) -> str:
        """
        Name sanitized for use in python code

        Returns:
            str: sanitized name
        """
        return sanitize(self.name)
    
    @property
    def descriptor(self) -> str:
        """
        Returns a string that describes the resonance
        """
        return f"{self.sanitized_name}_to_{'_'.join([d.sanitized_name for d in self.daughters])}"

    @property
    def id(self) -> int:
        return self.__id
    
    @id.setter
    def id(self, value:int):
        raise ValueError("The id of a resonance cannot be changed")

    @classmethod
    def register(cls, obj) -> int:
        instance_id = len(cls.__instances)
        cls.__instances[instance_id] = obj
        if obj.__name is not None:
            if not obj.__name in cls.__named_instances:
                # We have multiple resonances with the same name, we assume that they share a parameter set 
                # and thus we only need to register the name once
                # The parameter set is only for the lineshape and not for the couplings
                cls.__named_instances[obj.__name] = [obj]
            else:
                cls.__named_instances[obj.__name].append(obj)

        return instance_id
    
    def copy(self):
        return Resonance(self.node, quantum_numbers=self.quantum_numbers, lineshape=self.__lineshape, argnames=self.__parameter_names, name=self.__name, preserve_partity=self.preserve_partity, scheme=self.__scheme)

    @property
    def lineshape(self) -> Callable:
        return self.__lineshape

    @property
    def tuple(self) -> tuple:
        return self.node.value
    
    @property
    def daughter_qn(self) -> list[QN]:
        if self.daughters is None:
            raise ValueError(f"{self.node}: Daughter quantum numbers not set! This should happen in the DecayChainNode class. The resonance class should only be used as part of a DecayChain!")
        return [daughter.quantum_numbers for daughter in self.__daughters]
    

    @property   
    def daughters(self) ->  list["Resonance" | Node]:
        if self.__daughters is None:
            raise ValueError(f"{self.node}: Daughters not set! This should happen in the DecayChainNode class. The resonance class should only be used as part of a DecayChain!")
        return self.__daughters

    @daughters.setter
    def daughters(self, daughters: list[Resonance | Node]):
        self.__daughters = daughters
    
    def __str__(self):
        return f"Resonance {self.name} at {self.node} with {self.quantum_numbers}"
    
    def __repr__(self):
        return self.__str__()
    
    @convert_angular
    def helicity_from_ls(self, h0:Union[Angular, int], h1:Union[Angular, int], h2:Union[Angular, int], couplings:dict[LSTuple, float], arguments:dict):
        """
        This function translates from the ls basis into the helicity basis.
        The linehspae funcitons can depend on L and S.
        It is important, that particle 2 convention from Jacob-Wick is used!
        Otherwise the direct mapping of LS to helicity does not work.

        Note:
            Wrapper convert_angular ensures, that inside the function only integes arrive as values for h.

        arguments:
        h0: int
            Helicity of the resonance
        h1: int
            Helicity of the first daughter
        h2: int
            Helicity of the second daughter
        couplings: dict
            The couplings of the resonance.
            Format is {(l, s): value}
        arguments: dict
            The arguments for the lineshape function. The keys are the names of the arguments.
        
        """
        q1, q2 = self.daughter_qn
        j1, j2 = q1.angular.value2, q2.angular.value2

        return sum(
            coupling * 
            self.lineshape(l,s, *self.argument_list(arguments)) * 
            (l + 1) ** 0.5 /
            (self.quantum_numbers.angular.value2 + 1) ** 0.5 *
            clebsch_gordan(j1, h1, j2, -h2, s, h1- h2) *
            clebsch_gordan(l, 0, s, h1 - h2, self.quantum_numbers.angular.value2, h1 - h2)
            for (l, s), coupling in couplings.items()
        )


    def __construct_couplings(self, arguments:dict) -> dict[LSTuple, float]:
        """
        TEMPORARY VERSION FINAL SOLUTION NOT YET CLEAR

        We constuct the couplings for the resonance from real numbers found in the arguments dict
        Thus we need to know, which arguments belong to the couplings of the resonance
        The arguments are a dict of the form {id: {parameter_name: value}} where the id is the id of the resonance
        The ls couplings are a dict of the form {(l,s): value_r } under the name ls_couplings
        """
        couplings = arguments[self.id]["couplings"]
        return {LSTuple(*key): value for key, value in couplings.items()}

    def generate_couplings(self, conserve_parity:bool = True) -> dict[Union[LSTuple, HelicityTuple], float]:
        """
        This function should be used to generate the couplings for the resonance
        """

        if self.scheme == "helicity":
            return {
                "couplings": {
                    HelicityTuple(h1, h2): 1
                    for h1 in self.daughter_qn[0].projections(return_int=True)
                    for h2 in self.daughter_qn[1].projections(return_int=True)
                }
            }
        qn1, qn2 = self.daughter_qn
        qn0 = self.quantum_numbers

        if qn0.angular.value2 % 2 != (qn1.angular.value2 + qn2.angular.value2) % 2:
            raise ValueError(f"Angular momentum of resonance {self.name} at {self.node} is not compatible with the angular momenta of the daughters {qn1}, {qn2}.")
        possible_states = set(QN.generate_L_states(qn0, qn1, qn2))
        if not conserve_parity:
            qn0_bar = QN(qn0.angular.angular_momentum, -qn0.parity)
            possible_states = possible_states.union(set(QN.generate_L_states(qn0_bar, qn1, qn2)))
        if len(possible_states) == 0:
            raise ValueError(f"No possible states for resonance {self.name} at {self.node} with {qn0}, {qn1}, {qn2}.")
        return {
                "couplings": {
                    LSTuple(l.value2, s.value2): 1
                    for l, s in possible_states
                }
            }
    
    def direct_helicity_coupling(self, arguments, h1, h2):
        return arguments[self.id]["couplings"][(h1, h2)] * self.lineshape(h1, h2, *self.argument_list(arguments))
    
    @convert_angular
    def amplitude(self, h0:Union[Angular, int], h1:Union[Angular, int], h2:Union[Angular, int], arguments:dict):
        if self.scheme == "ls":
            couplings = self.__construct_couplings(arguments)
        # lineshape_args = self.__construct_arguments(arguments)
            coupling = self.helicity_from_ls(h0, h1, h2, couplings ,arguments)
        elif self.scheme == "helicity":
            coupling = self.direct_helicity_coupling(arguments, h1, h2)
        else:
            raise ValueError(f"Scheme must be either 'ls' or 'helicity' but is {self.scheme}")
        # particle 2 convention from Jacob-Wick is used!
        j2 = self.daughter_qn[1].angular.value2
        return coupling * (-1) ** ((j2 - h2) / 2) 
    
    def register_lineshape(self, lineshape_function:Callable, parameter_names: list[str]):
        if self.__lineshape is not None:
            raise ValueError("Lineshape already set")
        self.__lineshape = lineshape_function
        self.__parameter_names = {}
        for parameter_name in parameter_names:
            if parameter_name not in self.__parameter_names:
                type(self).__parameter_names[parameter_name] = self
        return self