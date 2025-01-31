from typing import Union

from decayangle.decay_topology import Topology, HelicityAngles, WignerAngles
from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN, wigner_capital_d, Angular, convert_angular
import numpy as np

class DecayChainNode:
    def __init__(self, tuple_value, resonances: dict[tuple, Resonance], final_state_qn: dict[tuple, QN],helicity_angles:dict, topology:Topology, convention:str="helicity") -> None:
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
        self.resonance = resonances.get(tuple_value, None)
        del tuple_value # explicitly delete this value after here

        self.resonances = resonances
        self.topology = topology
        self.final_state_qn = final_state_qn
        self.helicity_angles = helicity_angles
        self.convention = convention
            
        self.daughters = [
                    DecayChainNode(daughter.tuple, resonances, self.final_state_qn, helicity_angles, topology)
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
        self.helicity_angles = topology.helicity_angles(momenta=momenta)
        self.final_state_qn = final_state_qn
        self.nodes

        # we need a sorted version of the particle keys to map matrix elements to the correct particle helicities later
        self.final_state_keys = sorted(final_state_qn.keys())
        helicities = Angular.generate_helicities(*[final_state_qn[key] for key in self.final_state_keys])
        self.helicities = [
            {key: helicity[i] for i, key in enumerate(self.final_state_keys)}
            for helicity in helicities
        ]
        self.helicity_tuples = helicities
    
    @property
    def nodes(self):
        return list(
            DecayChainNode(node.tuple, self.resonances, self.final_state_qn, self.helicity_angles, self.topology)
            for node in self.topology.nodes.values()
        )

    @property
    def root(self):
        return DecayChainNode(self.topology.root.tuple, self.resonances, self.final_state_qn, self.helicity_angles, self.topology)

    @property
    def chain_function(self):

        def f(h0, lambdas:dict, helicity_angles:dict[tuple, HelicityAngles], arguments:dict):
            amplitudes = [
                 amplitude
                for amplitude in self.root.amplitude(h0, lambdas, helicity_angles, arguments)
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
        def matrix(h0, helicity_angles:dict[tuple, HelicityAngles], arguments:dict):
            return {
                tuple([lambdas[key] for key in self.final_state_keys]): f(h0, lambdas, helicity_angles, arguments)
                for lambdas in self.helicities
            }
        
        return matrix

class AlignedChain(DecayChain):
    
    def __init__(self, topology:Topology, resonances: dict[tuple, Resonance], momenta: dict, final_state_qn: dict[tuple, QN], reference:Topology | DecayChain, wigner_rotation: dict[tuple, WignerAngles]= None) -> None:
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
        def f(h0, helicity_angles:dict[tuple, HelicityAngles], arguments:dict):
            matrix = m(h0, helicity_angles, arguments)
            aligned_matrix = {
                self.to_tuple(lambdas): sum(
                    matrix[self.to_tuple(lambdas_)]
                    * np.prod([
                            self.wigner_dict[key][(lambdas[key], lambdas_[key])] for key in self.final_state_keys
                        ], 
                        axis=0)
                    for lambdas_ in self.helicities
                )
                for lambdas in self.helicities
            }
            return aligned_matrix
        
        return f



