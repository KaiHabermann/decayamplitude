from decayangle.decay_topology import Topology
from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN

class DecayChainNode:
    def __init__(self, tuple_value, resonances: dict[tuple, Resonance], final_state_qn: dict[tuple, QN],helicity_angles:dict, topology:Topology) -> None:
        # this check needs to happen first to avoid errors
        if tuple_value not in topology.nodes:
            if  all(t in topology.nodes for t in tuple_value):
                # if we are root, the above may be the case
                self.tuple = 0
                tuple_value = 0
            else:
                raise ValueError(f"Node {tuple_value} not in topology")

        self.tuple = tuple_value
        self.resonances = resonances
        self.resonance = resonances.get(tuple_value, None)
        self.topology = topology
        self.final_state_qn = final_state_qn
        self.helicity_angles = helicity_angles
            
        self.node = topology.nodes[tuple_value]
        self.daughters = [
                    DecayChainNode(daughter.tuple, resonances, self.final_state_qn, helicity_angles, topology)
                    for daughter in self.node.daughters
            ]
        
        if not self.final_state:
            # set the daughters of the resonance
            self.quantum_numbers = self.resonance.quantum_numbers
            self.resonance.daughter_qn = [daughter.quantum_numbers for daughter in self.daughters]
        else:
            self.quantum_numbers = self.final_state_qn[self.tuple]

    @property
    def final_state(self):
        return self.node.final_state

    @property
    def quantum_numbers(self) -> QN:
        return self.__qn 
    
    @quantum_numbers.setter
    def quantum_numbers(self, qn: QN):
        self.__qn = qn
    
    def amplitude(self, h0, lambdas:dict, arguments:dict):
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
                d1_helicities = d1.quantum_numbers.angular.projections()
            if d2.final_state:
                d2_helicities = [lambdas[d2.tuple]]
            else:
                d2_helicities = d2.quantum_numbers.angular.projections()

            for h1 in d1_helicities:
                for h2 in d2_helicities:
                    for A_1 in d1.amplitude(h1, lambdas, arguments):
                        for A_2 in d2.amplitude(h2, lambdas, arguments):
                            # TODO: add explicit handling of the arguments for the lineshape
                            yield A_1 * A_2 * self.resonance.amplitude(h0, h1, h2, arguments)

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
    
    @property
    def nodes(self):
        return list(self.topology.nodes.values())

    @property
    def root(self):
        return DecayChainNode(self.topology.root.tuple, self.resonances, self.final_state_qn, self.helicity_angles, self.topology)

    @property
    def chain_function(self):

        def f(h0, lambdas:dict, arguments:dict):
            return sum(
                amplitude
                for amplitude in self.root.amplitude(h0, lambdas, arguments)
            )

        return f
