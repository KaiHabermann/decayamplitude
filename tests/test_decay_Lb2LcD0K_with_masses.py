
# Test that daughter masses are correctly passed to lineshape functions

from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN
from decayamplitude.chain import MultiChain
from decayamplitude.combiner import ChainCombiner
from decayamplitude.particle import Particle, DecaySetup
from decayamplitude.backend import numpy as np
from decayangle.decay_topology import Topology, Node
from decayangle.config import config as decayangle_config

import numpy as onp

decayangle_config.backend = "numpy"
decayangle_config.sorting = "value"

topologies = [
    Topology(0, decay_topology=((2, 3), 1)),
    Topology(0, decay_topology=((1, 3), 2))
]

final_state_qn = {
    1: Particle(spin=1, parity=1, name="Lambda_c()+"),
    2: Particle(spin=0, parity=-1, name="Dbar0"),
    3: Particle(spin=0, parity=-1, name="K-"),
}
setup = DecaySetup(final_state_particles=final_state_qn)
topologies = [
    setup.symmetrize(topology) for topology in topologies
]


def constant_lineshape(L, S, *args):
    return 1.0


def make_recording_lineshape(store: dict):
    """Returns a BW lineshape that records the daughter masses it receives."""
    def lineshape(l, s, m0, gamma, *, d1_mass=None, d2_mass=None):
        store["d1"].append(d1_mass)
        store["d2"].append(d2_mass)
        mass = d1_mass if d1_mass is not None else 1.0
        return 1 / (mass**2 - m0**2 + 1j * mass * gamma)
    return lineshape


def amplitude(momenta, resonance_lineshapes: dict):
    resonances = {
        0: [
            Resonance(Node(0), quantum_numbers=QN(1, 1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name='Lambda_b()0')
        ],
        (2, 3): [
            Resonance(Node((2, 3)), quantum_numbers=QN(2, -1), lineshape=resonance_lineshapes[(2, 3)][0], argnames=["m0", "gamma"], preserve_partity=True, name='D_s_Kmatrix'),
            Resonance(Node((2, 3)), quantum_numbers=QN(0, 1), lineshape=resonance_lineshapes[(2, 3)][1], argnames=["m0", "gamma"], preserve_partity=True, name='D_s0^*(2317)-')
        ],
        (1, 3): [
            Resonance(Node((1, 3)), quantum_numbers=QN(3, -1), lineshape=resonance_lineshapes[(1, 3)][0], argnames=["m0", "gamma"], preserve_partity=True, name='Xi_c(2815)0'),
            Resonance(Node((1, 3)), quantum_numbers=QN(1, -1), lineshape=resonance_lineshapes[(1, 3)][1], argnames=["m0", "gamma"], preserve_partity=True, name='Xi_c(2923)')
        ]
    }

    chains = [
        MultiChain(
            topology=topology,
            resonances=resonances,
            momenta=momenta,
            final_state_qn=final_state_qn
        ) for topology in topologies
    ]
    return ChainCombiner(chains)


def test_Lb2LcD0K_masses_passed_to_lineshape():
    import json

    with open("tests/test_data/Lb2LcD0K_momenta.json", "r") as f:
        momenta = json.load(f)
        momenta = {int(k): onp.array(v) for k, v in momenta.items()}

    store_23_0 = {"d1": [], "d2": []}
    store_23_1 = {"d1": [], "d2": []}
    store_13_0 = {"d1": [], "d2": []}
    store_13_1 = {"d1": [], "d2": []}

    lineshapes = {
        (2, 3): [make_recording_lineshape(store_23_0), make_recording_lineshape(store_23_1)],
        (1, 3): [make_recording_lineshape(store_13_0), make_recording_lineshape(store_13_1)],
    }

    full = amplitude(momenta, lineshapes)
    couplings = full.generate_couplings()
    unpolarized, param_names = full.unpolarized_amplitude(couplings)

    start_params = {name: 1.0 for name in param_names}
    # run without JIT so lineshapes are called with concrete values
    unpolarized(**start_params)

    for store, label in [
        (store_23_0, "D_s_Kmatrix (2,3)"),
        (store_23_1, "D_s0^*(2317)- (2,3)"),
        (store_13_0, "Xi_c(2815)0 (1,3)"),
        (store_13_1, "Xi_c(2923) (1,3)"),
    ]:
        assert len(store["d1"]) > 0, f"{label}: d1_mass was never passed"
        assert len(store["d2"]) > 0, f"{label}: d2_mass was never passed"
        assert all(m is not None for m in store["d1"]), f"{label}: d1_mass contained None"
        assert all(m is not None for m in store["d2"]), f"{label}: d2_mass contained None"
        assert all(onp.all(onp.isfinite(m)) for m in store["d1"]), f"{label}: d1_mass contained non-finite values"
        assert all(onp.all(onp.isfinite(m)) for m in store["d2"]), f"{label}: d2_mass contained non-finite values"
        assert all(onp.all(m > 0) for m in store["d1"]), f"{label}: d1_mass contained non-positive values"
        assert all(onp.all(m > 0) for m in store["d2"]), f"{label}: d2_mass contained non-positive values"


if __name__ == "__main__":
    test_Lb2LcD0K_masses_passed_to_lineshape()
