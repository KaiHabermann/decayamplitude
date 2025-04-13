from __future__ import annotations
from decayamplitude.rotation import QN
from decayamplitude.chain import MultiChain
from decayamplitude.combiner import ChainCombiner
from decayamplitude.resonance import Resonance
from decayangle.decay_topology import Topology, Node

import numpy as np
def constant_lineshape(*args):
    return 1.0


def test_multi_chain():
    momenta = {
        1: np.array([1, 0.1, 0.4, 3]),
        2: np.array([0.5, -0.1, -0.4, 3]),
        3: np.array([1.1, 0.2, 0.5, 3]),
        4: np.array([0.6, -0.2, -0.5, 3]),
    }
    final_state_qn = {
            1: QN(0, 1), 
            2: QN(0, 1), 
            3: QN(1, 1), 
            4: QN(1, -1) 
        }

    resonances_hadronic = {
        (1,2): [
            # Here the hadronic resonances go+
            # These will decay strong, so we need to conserve parity
            Resonance(Node((1, 2)), quantum_numbers=QN(0, 1), lineshape=constant_lineshape, argnames=["D_2300_M", "D_2300_Gamma"], preserve_partity=True, name="Resnance1"),
            Resonance(Node((1, 2)), quantum_numbers=QN(4, 1), lineshape=constant_lineshape, argnames=["D_2460_M", "D_2460_Gamma"], preserve_partity=True, name="Resonance2"),
        ],
        (3, 4): 
        [Resonance(Node((3, 4)), quantum_numbers=QN(2, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="Resonance3")],
        (1,2,3): 
        [Resonance(Node((1, 2, 3)), quantum_numbers=QN(1, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="Resonance4")],

        0: [Resonance(Node(0), quantum_numbers=QN(0, 1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="B0")],
    }

    topology1 = Topology(
        0,
        decay_topology=((1,2), (3, 4))
    )

    momenta = topology1.to_rest_frame(momenta)

    topology2 = Topology(
        0,
        decay_topology=(((1,2), 3) ,4 )
    )

    chain1 = MultiChain(
        topology = topology1,
        resonances = resonances_hadronic,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    chain2 = MultiChain(
        topology = topology2,
        resonances = resonances_hadronic,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    assert len(chain1.chains) == 2
    for chain in chain1.chains:
        # root resonance + 2 chain resonances
        assert len(chain.resonances) == 3

    assert len(chain2.chains) == 2
    for chain in chain2.chains:
        assert len(chain.resonances) == 3

    combined = ChainCombiner([chain1, chain2])
    func, params = combined.unpolarized_amplitude(combined.generate_couplings())





if __name__ == "__main__":
    test_multi_chain()