import numpy as np
import decayamplitude
from decayamplitude.chain import DecayChain
from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN

from decayangle.decay_topology import Topology, Node

def constant_lineshape(*args):
    return 1

def test_threebody_1():
    topology = Topology(
        0,
        decay_topology=(1,(2,3))
    )

    resonances = {
        (2,3): Resonance(Node(2, 3), 2, -1, lineshape=constant_lineshape, argnames=[]),
        0: Resonance(Node(0), 0, 1, lineshape=constant_lineshape, argnames=[])

    }

    momenta = {
        1: np.array([1, 0, 0, 1.5]),
        2: np.array([-1, 0.5, 0, 2]),
        3: np.array([0, -0.5, 0, 0])
    }

    decay = DecayChain(
        topology = topology,
        resonances = resonances,
        momenta = momenta,
        final_state_qn = {
            1: QN(0, 1),
            2: QN(0, 1),
            3: QN(0, 1)
        }
    )

    print(decay.chain_function(0, lambdas={1:0, 2:0, 3:0}, arguments={}))

test_threebody_1()