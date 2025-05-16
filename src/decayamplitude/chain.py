from typing import Union, Optional, Callable, Literal
from itertools import product
from functools import cached_property

from decayangle.decay_topology import Topology, Node

from decayangle.decay_topology import Topology, HelicityAngles, WignerAngles
from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN, wigner_capital_d, Angular, convert_angular
from decayamplitude.backend import numpy as np

from decayamplitude.utils import _create_function, sanitize

class DecayChainNode:
    """
    Class to represent a node in the decay chain. This utilizes the Node class from decayangle. 
    A Node has a resonance, and a topology, to make senese of its position in the decay chain. The node value only has a meaning in the context of the topology.
    """


    def __init__(self, node:Node, resonances: dict[tuple, Resonance], final_state_qn: dict[tuple, QN], topology:Topology, convention:Literal["helicity", "minus_phi"]="helicity") -> None:
        """
        Initializes a DecayChainNode object. The object will contain a resonance and a topology.

        Parameters:
        node: Node
            The node of the decay chain. This is a node of the decay topology as defined in `decayangle`
        resonances: dict[tuple, Resonance]
            A dictionary with the resonances of the decay chain. The keys are the tuples of the nodes, the values are the resonances
        final_state_qn: dict[tuple, QN]
            A dictionary with the quantum numbers of the final state particles. The keys are the tuples of the nodes, the values are the quantum numbers
        topology: Topology
            The topology of the decay chain. This is a topology as defined in `decayangle`
        convention: str
            The convention of the decay chain. This is either "helicity" or "minus_phi". The default is "helicity"
        """
        # this check needs to happen first to avoid errors

        if node.value not in topology.nodes:
            # someone may have initialized the root with a tuple instead of 0
            if node.tuple == topology.root.tuple:
                self.node = topology.root
            else:
                raise ValueError(f"Node {node} not in topology {topology}")
        else:
            self.node = topology.nodes[node.value]
        
        self.tuple = self.node.tuple
        self.__is_root = self.node == topology.root
            
        self.resonance: Union[Resonance, None] = resonances.get(self.tuple, resonances.get(self.node.value, None))
        if self.resonance is None and not self.final_state:
            raise ValueError(f"Resonance for {self.node.tuple} not found. Every internal node must have a resonance to describe its behaviour!")

        self.resonances = resonances
        self.topology = topology
        self.final_state_qn = final_state_qn
        self.convention = convention
            
        self.daughters = [
                    DecayChainNode(daughter, resonances, self.final_state_qn, topology, convention=self.convention)
                    for daughter in self.node.daughters
            ]
        
        if not self.final_state:
            if self.resonance is None:
                raise ValueError(f"Resonance for {self.tuple} not found. Every internal node must have a resonance to describe its behaviour!")
            # set the daughters of the resonance
            self.quantum_numbers = self.resonance.quantum_numbers
            self.resonance.daughters = [daughter for daughter in self.daughters]
        else:
            self.quantum_numbers = self.final_state_qn[self.tuple]

    @property
    def final_state(self):
        """
        Returns:
        bool
            True if the node is a final state particle, False otherwise
        """
        return self.node.final_state
    
    @property
    def name(self) -> str:
        """
        Returns:
        str
            The name of the node
        """
        if self.resonance is not None:
            return self.resonance.name
        return f"particle_{self.node.value}"
    
    @property
    def sanitized_name(self) -> str:
        """
        Returns:
        str
            The sanitized name of the node
        """
        if self.resonance is not None:
            return self.resonance.sanitized_name
        return sanitize(f"particle_{self.node.value}")
    
    @property
    def is_root(self):
        """
        Returns:
        bool
            True if the node is the root of the decay chain, False otherwise
        """
        return self.__is_root

    @property
    def quantum_numbers(self) -> QN:
        """
        Returns:
        QN
            The quantum numbers of the node
        """
        return self.__qn 
    
    @property
    def decay_tuple(self) -> tuple:
        """
        Returns:
        tuple
            The decay tuple of the node. This is the tuple of the nodes daughters values.
        """
        return tuple([daughter.node.value for daughter in self.daughters])
    
    @quantum_numbers.setter
    def quantum_numbers(self, qn: QN):
        """
        Sets the quantum numbers of the node. This is used to set the quantum numbers of the resonance.
        """
        self.__qn = qn

    def __helicity_angles(self, angles:HelicityAngles) -> tuple:
        """
        Returns the helicity angles of the node. This is used to calculate the amplitude of the decay chain.

        Parameters:
            angles: HelicityAngles
                The helicity angles of the node. This is used to calculate the amplitude of the decay chain. Helicity angles are defined in the `decayangle` library.
        Returns:
            tuple
                The helicity angles of the node. This is used to calculate the amplitude of the decay chain.
        """
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
                            A_self = self.resonance.amplitude(h0, h1, h2, arguments) * np.conj(wigner_capital_d(*self.__helicity_angles(helicity_angles[self.decay_tuple]), self.quantum_numbers.angular.value2, h0, h1 - h2))
                            yield A_1 * A_2 * A_self * (self.quantum_numbers.angular.value2 + 1)**0.5

class DecayChain:
    """
    Class to represent a decay chain. This is a topology in connection with a set of resonances. One resonance for each internal node in the topology.
    """

    def __init__(self, topology:Topology, resonances: dict[tuple, Resonance], momenta: dict, final_state_qn: dict[tuple, QN], convention:Literal["helicity", "minus_phi"]="helicity") -> None:
        self.topology = topology
        self.resonances = resonances
        self.momenta = momenta
        self.final_state_qn = final_state_qn
        self.convention = convention

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
            DecayChainNode(node, self.resonances, self.final_state_qn, self.topology, self.convention)
            for node in self.topology.nodes.values()
        )
    
    @property
    def final_state_nodes(self) -> list[DecayChainNode]:
        return [node for node in self.nodes if node.final_state]

    @cached_property
    def helicity_angles(self):
        return self.topology.helicity_angles(momenta=self.momenta, convention=self.convention)

    @property
    def root(self):
        return DecayChainNode(self.topology.root, self.resonances, self.final_state_qn, self.topology, self.convention)

    @property
    def chain_function(self):

        def f(h0, lambdas:dict, arguments:dict):
            amplitudes = [
                 amplitude
                for amplitude in self.root.amplitude(h0, lambdas, self.helicity_angles, arguments)
            ]
            prefactor = 1/(self.root.resonance.quantum_numbers.angular.value2 + 1)**0.5
            return prefactor * sum(
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
    
    def generate_couplings(self):
        """
        Returns all LS couplings for the decay chain
        """
        return {
            node.resonance.id: node.resonance.generate_couplings(node.resonance.preserve_partity)
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
    """
    The aligned version of the decay chain. This is used to calculate the aligned amplitude, which is the amplitude in the final state helicity frame as defined by a reference topology or reference chain.
    """

    def __init__(self, topology:Topology, resonances: dict[tuple, Resonance], momenta: dict, final_state_qn: dict[tuple, QN], reference:Union[Topology, DecayChain], wigner_rotation: dict[tuple, WignerAngles]= None, convention:Literal["helicity", "minus_phi"]="helicity") -> None:
        """
        Initializes an AlignedChain object. The object will contain a list of DecayChain objects.
        Parameters:
        topology: Topology
            The topology of the decay chain
        resonances: dict[tuple, Resonance]
            A dictionary with the resonances of the decay chain. The keys are the tuples of the nodes, the values are the resonances
        momenta: dict
            The momenta of the decay chain
        final_state_qn: dict[tuple, QN]
            A dictionary with the quantum numbers of the final state particles
        reference: Topology
            The reference topology for the decay chain. This is used to calculate the alignment.
        wigner_rotation: dict[tuple, WignerAngles]
            A dictionary with the Wigner angles for the decay chain. This is used to calculate the alignment. Calculated if not provided.
        convention: str
            The convention of the decay chain. This is either "helicity" or "minus_phi". The default is "helicity"
        """
        self.reference: Topology = reference if isinstance(reference, Topology) else reference.topology
        self.topology = topology

        super().__init__(topology, resonances, momenta, final_state_qn, convention)
        if wigner_rotation is None:
            if isinstance(reference, DecayChain):
                if reference.convention != convention:
                    raise ValueError(f"Reference and chain must have the same convention. Found reference: {reference.convention} and self: {convention}!")
            self.wigner_rotation = self.reference.relative_wigner_angles(self.topology, momenta, convention=self.convention)
        else:
            self.wigner_rotation = wigner_rotation
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
    
    def aligned_matrix_function(self, couplings:dict) -> Callable:
        """
        Returns a function, which will return the amplitude for a given set of helicities. 
        The function will use the matrix to perform the calculation.
        """
        aligned_matrix = self.aligned_matrix
        def f(arguments: dict):
            h0 = arguments.pop("h0")
            return aligned_matrix(h0, arguments)

        return _create_function(
            ["h0"] + self.resonance_params, couplings, f
        )

    
class MultiChain(DecayChain):
    @classmethod
    def create_chains(cls, resonances: dict[tuple, tuple[Resonance]], topology: Topology | None = None) -> list[dict[tuple, Resonance]]:
        """
        Creates all possible chains from a dictionary with lists of reonances for each isobar
        """
        if topology is not None:
            # with a given topology we can restrict the resonances to the nodes in the topology
            # this is usefull, if we only have one global dict of resonances
            all_nodes = [node for node in topology.nodes]
            resonances = {k: v for k, v in resonances.items() if k in all_nodes}
        ordered_keys = list(resonances.keys())
        chains = product(*[resonances[key] for key in ordered_keys])
        return [
            {
                key: chain[i].copy()
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

    def __init__(self, topology:Topology, momenta: dict, final_state_qn: dict[tuple, QN], resonances: Optional[dict[tuple, tuple[Resonance]]]=None, chains: Optional[list[DecayChain]]=None, convention:Optional[Literal["minus_phi", "helicity"]]="helicity") -> None:
        """
        Initializes a MultiChain object. The object will contain a list of DecayChain objects.

        Parameters:
        topology: Topology
            The topology of the decay chain
        momenta: dict
            The momenta of the decay chain
        final_state_qn: dict[tuple, QN]
            The quantum numbers of the final state particles
        resonances: dict[tuple, tuple[Resonance]]
            A dictionary with a list of resonances for each isobar
        chains: list[DecayChain]
            A list of DecayChain objects
        convention: str
            The convention of the decay chain. Default is "helicity"
        """

        if chains is not None:
            self.chains = chains
            # I will stick with a default value for the convention. It will have no effect if chains are provided
            # if convention is not None:
            #     raise ValueError("Convention must not be set if chains are provided directly")
            if not all(chain.convention == chains[0].convention for chain in chains):
                raise ValueError("All chains must have the same convention")
            self.convention = chains[0].convention
        elif resonances is not None:
            resonant_nodes = [node for node in topology.nodes.values() if not node.final_state]
            if any(node.value not in resonances and node.tuple not in resonances for node in resonant_nodes):
                raise ValueError(f"Not all nodes have a resonance assigned: {resonances.keys()}, {list(map(lambda x: x.value,resonant_nodes))}")
            self.chains = [
                DecayChain(topology, chain_definition, momenta, final_state_qn, convention)
                for chain_definition in type(self).create_chains(resonances, topology)
            ]
            def chain_filter(chain):
                try:
                    chain.generate_couplings()
                except ValueError:
                    return False
                return True
            self.chains = [chain for chain in self.chains if chain_filter(chain)]
            if not self.chains:
                raise ValueError("There are no valid chains in the provided resonances! Check the resonances and the topology! Or check the quantum numbers!")
            self.convention = convention
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
    
    def generate_couplings(self):
        """
        Returns all LS couplings for the decay chain
        """
        coupling_dict = {}
        for chain in self.chains:
            coupling_dict.update(chain.generate_couplings())
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

    def __init__(self, topology:Topology, momenta: dict, final_state_qn: dict[tuple, QN], reference:Union[Topology, DecayChain], resonances: Optional[dict[tuple, tuple[Resonance]]] = None, chains: Optional[list[DecayChain]] = None, wigner_rotation: dict[tuple, WignerAngles]= None, convention:Literal["helicity", "minus_phi"]="helicity") -> None:
        super().__init__(topology, momenta, final_state_qn, resonances=resonances, chains=chains, convention=convention)
        self.reference: Topology = reference if isinstance(reference, Topology) else reference.topology
        if wigner_rotation is None:
            self.wigner_rotation = self.reference.relative_wigner_angles(self.topology, momenta, convention=self.convention)
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





