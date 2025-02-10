from typing import Union, Optional, Callable
from itertools import product
from functools import cached_property

from decayangle.decay_topology import Topology, HelicityAngles, WignerAngles
from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN, wigner_capital_d, Angular, convert_angular
from decayamplitude.backend import numpy as np

class DecayChainNode:
    def __init__(self, tuple_value, resonances: dict[tuple, Resonance], final_state_qn: dict[tuple, QN], topology:Topology, convention:str="helicity") -> None:
        # this check needs to happen first to avoid errors
        if tuple_value not in topology.nodes:
            if  all(t in topology.nodes for t in tuple_value):
                # if we are root, the above will be the case
                self.tuple = topology.root.tuple
                tuple_value = topology.root.value
                self.__is_root = True
            else:
                raise ValueError(f"Node {tuple_value} not in topology")
        else:
            self.tuple = tuple_value
            self.__is_root = False
        self.node = topology.nodes[tuple_value]
        self.resonance: Union[Resonance, None] = resonances.get(tuple_value, None)
        if self.resonance is None and not self.final_state:
            raise ValueError(f"Resonance for {tuple_value} not found. Every internal node must have a resonance to describe its behaviour!")
        del tuple_value # explicitly delete this value after here

        self.resonances = resonances
        self.topology = topology
        self.final_state_qn = final_state_qn
        self.convention = convention
            
        self.daughters = [
                    DecayChainNode(daughter.tuple, resonances, self.final_state_qn, topology)
                    for daughter in self.node.daughters
            ]
        
        if not self.final_state:
            if self.resonance is None:
                raise ValueError(f"Resonance for {tuple_value} not found. Every internal node must have a resonance to describe its behaviour!")
            # set the daughters of the resonance
            self.quantum_numbers = self.resonance.quantum_numbers
            self.resonance.daughter_qn = [daughter.quantum_numbers for daughter in self.daughters]
        else:
            self.quantum_numbers = self.final_state_qn[self.tuple]

    @property
    def final_state(self):
        return self.node.final_state
    
    @property
    def is_root(self):
        return self.__is_root

    @property
    def quantum_numbers(self) -> QN:
        return self.__qn 
    
    @quantum_numbers.setter
    def quantum_numbers(self, qn: QN):
        self.__qn = qn

    def __helicity_angles(self, angles:HelicityAngles) -> tuple:
        if self.convention == "helicity":
            return (angles.phi_rf, angles.theta_rf, 0)
        if self.convention == "minus_phi":
            return (angles.phi_rf, angles.theta_rf, -angles.phi_rf)
        raise ValueError(f"Convention {self.convention} not known")

    @convert_angular
    def amplitude(self, h0:Union[Angular, int], lambdas:dict, helicity_angles:dict[tuple,HelicityAngles], arguments:dict):
        """
        The amplitude of a single node given the helicity of the decaying particle
        The helicities of the daughters will be generated from here, recursively
        the arguments are the couplings of the resonances and are a global dict, which will be passed through the recursion
        This means, that all arguments for lineshapes will be provided positional 

        parameters:
        h0: int
            The helicity of the decaying particle
        lambdas: dict
            The helicites of the mother and the final state particles
        arguments: dict
            The couplings of the resonances and the resonance parameters
        
        returns:
        float
            The amplitude of the decay as a generator

        """
        if self.final_state:
            # we do not have a resonance and we have no daughters
            yield 1.
        else:
            d1, d2 = self.daughters
            if d1.final_state:
                d1_helicities = [lambdas[d1.tuple]]
            else:
                d1_helicities = d1.quantum_numbers.projections(return_int=True)
            if d2.final_state:
                d2_helicities = [lambdas[d2.tuple]]
            else:
                d2_helicities = d2.quantum_numbers.projections(return_int=True)

            for h1 in d1_helicities:
                for h2 in d2_helicities:
                    for A_1 in d1.amplitude(h1, lambdas, helicity_angles, arguments):
                        for A_2 in d2.amplitude(h2, lambdas, helicity_angles, arguments):
                            # TODO: add explicit handling of the arguments for the lineshape
                            A_self = self.resonance.amplitude(h0, h1, h2, arguments) * np.conj(wigner_capital_d(*self.__helicity_angles(helicity_angles[self.tuple]), self.quantum_numbers.angular.value2, h0, h1 - h2))
                            yield A_1 * A_2 * A_self * (self.quantum_numbers.angular.value2 + 1)**0.5

class DecayChain:
    """
    Class to represent a decay chain. This is a topology in connection with a set of resonances. One resonance for each internal node in the topology.
    """

    def __init__(self, topology:Topology, resonances: dict[tuple, Resonance], momenta: dict, final_state_qn: dict[tuple, QN]) -> None:
        self.topology = topology
        self.resonances = resonances
        self.momenta = momenta
        self.final_state_qn = final_state_qn
        self.nodes

        self.root_resonance = self.resonances.get(self.topology.root.value)
        if self.root_resonance is None:
            self.root_resonance = self.resonances.get(self.topology.root.tuple)
        if self.root_resonance is None:
            raise ValueError(f"No root resonance found! The root resonance should be the decaying particle. The lineshape is irrelelevant for decay processes, but the quantum numbers are crucial! Define a root resonance under the key {self.topology.root.value} or {self.topology.root.tuple}.")

        # we need a sorted version of the particle keys to map matrix elements to the correct particle helicities later
        self.final_state_keys = sorted(final_state_qn.keys())
        helicities = Angular.generate_helicities(*[final_state_qn[key] for key in self.final_state_keys])
        self.helicities = [
            {key: helicity[i] for i, key in enumerate(self.final_state_keys)}
            for helicity in helicities
        ]
        self.helicity_tuples = helicities
        self.resonance_list = list(resonances.values())
    
    @property
    def nodes(self):
        return list(
            DecayChainNode(node.tuple, self.resonances, self.final_state_qn, self.topology)
            for node in self.topology.nodes.values()
        )
    
    @property
    def final_state_nodes(self) -> list[DecayChainNode]:
        return [node for node in self.nodes if node.final_state]

    @cached_property
    def helicity_angles(self):
        return self.topology.helicity_angles(momenta=self.momenta)

    @property
    def root(self):
        return DecayChainNode(self.topology.root.tuple, self.resonances, self.final_state_qn, self.topology)

    @property
    def chain_function(self):

        def f(h0, lambdas:dict, arguments:dict):
            amplitudes = [
                 amplitude
                for amplitude in self.root.amplitude(h0, lambdas, self.helicity_angles, arguments)
            ]
            return sum(
               amplitudes
            )

        return f
    
    @property
    def matrix(self):
        """
        Returns a function, which will not produce the chain function for a single set of helicities, but will rather return a matrix with all possible helicities. 
        The matrix will be an actual matrix and not a dict, since we want to use it later to perform matrix operations.
        """

        f = self.chain_function
        def matrix(h0, arguments:dict):
            return {
                tuple([lambdas[key] for key in self.final_state_keys]): f(h0, lambdas, arguments)
                for lambdas in self.helicities
            }
        
        return matrix
    
    def generate_ls_couplings(self):
        """
        Returns all LS couplings for the decay chain
        """
        return {
            node.resonance.id: node.resonance.generate_ls_couplings(node.resonance.preserve_partity)
            for node in self.nodes
            if not node.final_state
        }
    
    @property
    def resonance_params(self):
        resonances = [resonance for resonance in self.resonance_list]
        resonance_parameter_names = [name for resonance in resonances for name in resonance.parameter_names]

        if len(set(resonance_parameter_names)) != len(resonance_parameter_names):
            from collections import Counter
            c = Counter(resonance_parameter_names)
            # raise ValueError(f"Parameter names are not unique: {', '.join([name for name, count in c.items() if count > 1])}")
        return list(set(resonance_parameter_names))

class AlignedChain(DecayChain):
    def __init__(self, topology:Topology, resonances: dict[tuple, Resonance], momenta: dict, final_state_qn: dict[tuple, QN], reference:Union[Topology, DecayChain], wigner_rotation: dict[tuple, WignerAngles]= None) -> None:
        self.reference: Topology = reference if isinstance(reference, Topology) else reference.topology
        self.topology = topology
        if wigner_rotation is None:
            self.wigner_rotation = self.reference.relative_wigner_angles(self.topology, momenta)
        else:
            self.wigner_rotation = wigner_rotation

        super().__init__(topology, resonances, momenta, final_state_qn)
        # we want the tuple versions of the helicities, since we use them as tuples
        self.wigner_dict = {
            key: {
                (h_, h): np.conj(wigner_capital_d(*self.wigner_rotation[key], final_state_qn[key].angular.value2, h, h_))
                for h in final_state_qn[key].angular.projections(return_int=True)
                for h_ in final_state_qn[key].angular.projections(return_int=True)
            }
            for key in self.final_state_keys
        }

    def to_tuple(self, lambdas:dict):
        return tuple([lambdas[key] for key in self.final_state_keys])

    @property
    def aligned_matrix(self):
        """
        Returns a function, which will return the amplitude for a given set of helicities. 
        The function will use the matrix to perform the calculation.
        """
        m = self.matrix
        def f(h0, arguments:dict):
            matrix = m(h0, arguments)
            aligned_matrix = {
                self.to_tuple(lambdas): sum(
                    matrix[self.to_tuple(lambdas_)]
                    * np.prod(np.array([
                            self.wigner_dict[key][(lambdas[key], lambdas_[key])] for key in self.final_state_keys
                        ]), 
                        axis=0)
                    for lambdas_ in self.helicities
                )
                for lambdas in self.helicities
            }
            return aligned_matrix
        
        return f
    
class MultiChain(DecayChain):
    @classmethod
    def create_chains(cls, resonances: dict[tuple, tuple[Resonance]]) -> list[dict[tuple, Resonance]]:
        """
        Creates all possible chains from a dictionary with lists of reonances for each isobar
        """
        ordered_keys = list(resonances.keys())
        chains = product(*[resonances[key] for key in ordered_keys])
        return [
            {
                key: chain[i]
                for i, key in enumerate(ordered_keys)
            }
            for chain in chains
        ]

    @classmethod
    def from_chains(cls, chains: list[DecayChain]) -> "MultiChain":
        new_obj = cls(chains[0].topology, chains[0].momenta, chains[0].final_state_qn, chains=chains)
        if any(chain.topology != chains[0].topology for chain in chains):
            raise ValueError("All chains must have the same topology")

        return new_obj

    def __init__(self, topology:Topology, momenta: dict, final_state_qn: dict[tuple, QN], resonances: Optional[dict[tuple, tuple[Resonance]]]=None, chains: Optional[list[DecayChain]]=None) -> None:
        if chains is not None:
            self.chains = chains
        elif resonances is not None:
            if any(node.value not in resonances and node.tuple not in resonances and not node.final_state for node in topology.nodes.values()):
                raise ValueError(f"Not all nodes have a resonance assigned: {resonances.keys()}, {topology.nodes.keys()}")
            self.chains = [
                DecayChain(topology, chain_definition, momenta, final_state_qn)
                for chain_definition in type(self).create_chains(resonances)
            ]
        else:
            raise ValueError("Either resonances or chains must be provided")
        if chains is not None and resonances is not None:
            raise ValueError("Either resonances or chains must be provided")

    @property
    def chain_function(self) -> Callable:
        def f(h0, lambdas:dict, arguments:dict):
            return sum(
                chain.chain_function(h0, lambdas, arguments)
                for chain in self.chains
            )
        return f
    
    @property
    def resonance_list(self) -> list[Resonance]:
        return [
            resonance for chain in self.chains
            for resonance in chain.resonance_list
        ]
    
    @property
    def final_state_keys(self) -> list[Union[tuple, int]]:
        return self.chains[0].final_state_keys
    
    @property
    def topology(self):
        return self.chains[0].topology

    @property
    def matrix(self):
        def dict_sum(*dtcs):
            if len(dtcs) == 1:
                return dtcs[0]
            if len(dtcs) == 0:
                raise ValueError("No dicts to sum")
            if any(set(dtcs[0].keys()) != set(dtc.keys()) for dtc in dtcs):
                raise ValueError("Keys of the dicts do not match")
            return {
                key: sum(dtc[key] for dtc in dtcs)
                for key in dtcs[0].keys()
            }

        def matrix(h0, arguments:dict):
            return dict_sum(
                *[chain.matrix(h0, arguments)
                for chain in self.chains]
            )
        return matrix
    
    @property
    def root(self):
        return self.chains[0].root
    
    @property
    def nodes(self):
        return self.chains[0].nodes
    
    @property
    def helicities(self):
        return self.chains[0].helicities
    
    @property
    def helicity_tuples(self):
        return self.chains[0].helicity_tuples
    
    def generate_ls_couplings(self):
        """
        Returns all LS couplings for the decay chain
        """
        coupling_dict = {}
        for chain in self.chains:
            coupling_dict.update(chain.generate_ls_couplings())
        return coupling_dict
    
    @property
    def root_resonance(self) -> Union[Resonance, None]:
        if all(chain.root_resonance.quantum_numbers == self.chains[0].root_resonance.quantum_numbers for chain in self.chains):
            return self.chains[0].root_resonance
        return None
    
class AlignedMultiChain(MultiChain):
    @classmethod
    def from_chains(cls, chains: list[DecayChain], reference:Union[Topology, DecayChain]) -> "AlignedMultiChain":
        return cls(
            chains[0].topology,
            chains[0].momenta,
            chains[0].final_state_qn,
            reference,
            chains=chains
        )

    @classmethod
    def from_multichain(cls, multichain: MultiChain, reference:Union[Topology, DecayChain]) -> "AlignedMultiChain":
        return cls.from_chains(
            multichain.chains,
            reference
        )

    def __init__(self, topology:Topology, momenta: dict, final_state_qn: dict[tuple, QN], reference:Union[Topology, DecayChain], resonances: Optional[dict[tuple, tuple[Resonance]]] = None, chains: Optional[list[DecayChain]] = None, wigner_rotation: dict[tuple, WignerAngles]= None) -> None:
        super().__init__(topology, momenta, final_state_qn, resonances=resonances, chains=chains)
        self.reference: Topology = reference if isinstance(reference, Topology) else reference.topology
        if wigner_rotation is None:
            self.wigner_rotation = self.reference.relative_wigner_angles(self.topology, momenta)
        else:
            self.wigner_rotation = wigner_rotation

        self.wigner_dict = {
            key: {
                (h_, h): np.conj(wigner_capital_d(*self.wigner_rotation[key], final_state_qn[key].angular.value2, h, h_))
                for h in final_state_qn[key].angular.projections(return_int=True)
                for h_ in final_state_qn[key].angular.projections(return_int=True)
            }
            for key in self.final_state_keys
        }

    def to_tuple(self, lambdas:dict):
        return tuple([lambdas[key] for key in self.final_state_keys])
        

    @property
    def aligned_matrix(self):
        """
        Returns a function, which will return the amplitude for a given set of helicities. 
        The function will use the matrix to perform the calculation.
        """
        m = self.matrix
        def f(h0,  arguments:dict):
            matrix = m(h0, arguments)
            aligned_matrix = {
                self.to_tuple(lambdas): sum(
                    matrix[self.to_tuple(lambdas_)]
                    * np.prod(np.array([
                            self.wigner_dict[key][(lambdas[key], lambdas_[key])] for key in self.final_state_keys
                        ]), 
                        axis=0)
                    for lambdas_ in self.helicities
                )
                for lambdas in self.helicities
            }
            return aligned_matrix
        
        return f





