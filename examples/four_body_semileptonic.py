from utils import make_four_vectors, constant_lineshape, BW_lineshape
from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN, Angular
from decayamplitude.chain import DecayChain, MultiChain
from decayamplitude.combiner import ChainCombiner

from decayamplitude.backend import numpy as np

from decayangle.decay_topology import Topology, Node
from decayangle.kinematics import mass
from decayangle.config import config as decayangle_config
decayangle_config.backend = "jax"

from jax import jit, grad

from collections import defaultdict

# Since we are semi leptonic there is only one important decay topology
# The (1, 2) system is the hadronic system, which is the one we are interested in
# Decay definition: B0 -> D0 h mu nu
topology1 = Topology(
    0,
    decay_topology=((1,2), (3, 4))
)

def resonances_BW(momenta):
    m_12 = topology1.nodes[(1,2)].mass(momenta=momenta)
    resonances_hadronic = {
        (1,2): [
            # Here the hadronic resonances go+
            # These will decay strong, so we need to conserve parity
            Resonance(Node((1, 2)), quantum_numbers=QN(0, 1), lineshape=BW_lineshape(m_12), argnames=["mass_resonance_1", "width_resonance_1"], preserve_partity=True),
            Resonance(Node((1, 2)), quantum_numbers=QN(0, 1), lineshape=BW_lineshape(m_12), argnames=["mass_resonance_2", "width_resonance_2"], preserve_partity=True),
            Resonance(Node((1, 2)), quantum_numbers=QN(0, 1), lineshape=BW_lineshape(m_12), argnames=["mass_resonance_3", "width_resonance_3"], preserve_partity=True),
            # Resonance(Node((1, 2)), quantum_numbers=QN(J, P), lineshape=BW_lineshape(m_12), argnames=["mass_resonance_n", "width_resonance_n"], preserve_partity=True), # template for further resonances
            ],
        # This is the W boson. It is defined as a resonance, but we assue a constant lineshape in this mass regime. One could use a more complicated one aswell.
        (3, 4): [Resonance(Node((3, 4)), quantum_numbers=QN(2, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False)],

        # This is the decaying B0 meson. It is defined as a resonance, but since this is a decay amplitude, the description is not important. Only the QN have to be correct. 
        0: [Resonance(Node(0), quantum_numbers=QN(0, 1), lineshape=constant_lineshape, argnames=[], preserve_partity=False)]
    }
    return resonances_hadronic


def phasespace_momenta():
    import phasespace
    B0_MASS = 5.27963
    PION_MASS = 0.13957018
    D0_MASS = 1.86483
    MU_MASS = 0.1056583715
    NU_MASS = 0
    weights, particles = phasespace.nbody_decay(B0_MASS,
                                            [D0_MASS, PION_MASS, MU_MASS, NU_MASS]).generate(n_events=10)
    return particles

def shortFourBodyAmplitudeBW():
    final_state_qn = {
        1: QN(0, 1), # D0
        2: QN(0, 1), # h
        3: QN(1, 1), # mu
        4: QN(1, -1) # nu
    }
    momenta = phasespace_momenta()
    momenta = {
        1: np.array(momenta["p_0"]),
        2: np.array(momenta["p_1"]),
        3: np.array(momenta["p_2"]),
        4: np.array(momenta["p_3"]),
    }

    # topology2 = Topology(
    #     0,
    #     decay_topology=((1, 2), 3)
    # )
    resonances_hadronic = resonances_BW(momenta)
    chain1 = MultiChain(
        topology = topology1,
        resonances = resonances_hadronic,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    # For semileptonics no actual alignment is needed, as we are only interested in one single decay chain
    # But doing it this way allows to use the convenience functions of the combiner
    full = ChainCombiner([chain1])
    unpolarized, argnames = full.unpolarized_amplitude(full.generate_ls_couplings())
    print(argnames)

    # an issue with jax, where the internal caching structure needs to be prewarmed, so that in the compilation step the correct types are inferred
    print(unpolarized(*([1] * len(argnames))))
    # we can now jit the function
    unpolarized = jit(unpolarized) 
    print(unpolarized(*([1] * len(argnames))))


    # for the gradient calculation we need to define a log likelihood function or something, that produces a single value
    def LL(*args):
        return np.sum(
            np.log(unpolarized(*args))
                )
    # we can calc the gradient of the log likelihood function
    unpolarized_grad = jit(grad(LL, argnums=[i for i in range(len(argnames))]))

    # and a test call
    print(unpolarized_grad(*([1.0] * len(argnames))))

    # polarized, lambdas ,polarized_argnames = full.polarized_amplitude(full.generate_ls_couplings())
    # print(lambdas)
    # lambda_values = [0, 0, 0, 1, 1]
    # print(polarized(*lambda_values,*([1] * len(polarized_argnames))) )

    # matrx_function, matrix_argnames = full.matrix_function(full.generate_ls_couplings())
    # print(matrix_argnames)
    # print(matrx_function(0, *([1] * len(argnames))) )


if __name__ == "__main__":
    shortFourBodyAmplitudeBW()