from decayangle.decay_topology import Topology
from decayamplitude.resonance import Resonance


class DecayChainNode:
    def __init__(self, name, resonances: dict[tuple, Resonance],helicity_angles:dict, topology:Topology) -> None:
        self.name = name
        self.resonances = resonances
        self.resonance = resonances[name]
        self.topology = topology
        self.helicity_angles = helicity_angles
        if name not in topology.nodes:
            raise ValueError(f"Node {name} not in topology")
        self.node = topology.nodes[name]
        self.daughter_resonances = [resonances[daughter] for daughter in self.node.daughters]

    def amplitude(self, h0, arguments:dict):
        if self.node.final_state:
            yield 1.
        else:
            d1, d2 = self.daughter_resonances
            for h1 in d1.quantum_numbers.angular.projections():
                for h2 in d2.quantum_numbers.angular.projections():
                    yield self.resonance.lineshape(**arguments) 


class DecayChain:
    """
    Class to represent a decay chain. This is a topology in connection with a set of resonances. One resonance for each internal node in the topology.
    """

    def __init__(self, topology:Topology, resonances: dict[tuple, Resonance], momenta: dict) -> None:
        self.topology = topology
        self.resonances = resonances
        self.momenta = momenta
        self.helicity_angles = topology.helicity_angles(momenta=momenta)
    
    @property
    def nodes(self):
        return list(self.topology.nodes.values())        

    @property
    def chain_function(self):

        def f(lambdas:dict, **kwargs):
            for node in self.nodes:
                node_function = self.resonances[node.tuple].lineshape

                # node_function = self.resonances[node].lineshape
                # node_function(*args)
    
