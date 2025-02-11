from utils import make_four_vectors, constant_lineshape, BW_lineshape
from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN, Angular
from decayamplitude.chain import DecayChain, MultiChain
from decayamplitude.combiner import ChainCombiner

from decayamplitude.backend import numpy as np

from decayangle.decay_topology import Topology, Node, TopologyCollection
from decayangle.kinematics import mass
from decayangle.config import config as decayangle_config
decayangle_config.backend = "jax"
decayangle_config.sorting = "off"

from jax import jit, grad

from collections import defaultdict

# Since we are semi leptonic there is only one important decay topology
# The (1, 2) system is the hadronic system, which is the one we are interested in
# Decay definition: B0 -> pip pim (J/psi -> mu+ mu-) 

# this will enforce, that we only get topologies, which have the intermediate state J/Psi -> mu+ mu-
tc = TopologyCollection(0, [1,2,3,4])

topology_pipi, = tc.filter((3,4), (1,2))
topology_pip_JPsi, = tc.filter((3,4), (1, 3, 4))
topology_pi_m_JPsi, = tc.filter((3,4), (2, 3, 4))
def resonances(momenta):
    m_pi_pi = topology_pipi.nodes[(1,2)].mass(momenta=momenta)
    resonances_pi_pi = {
        (1,2): [
            # Here the pi pi resonances go
            # These will decay strong, so we need to conserve parity
            Resonance(Node((1, 2)), quantum_numbers=QN(2, 1), lineshape=BW_lineshape(m_pi_pi), argnames=["mass_resonance_1", "width_resonance_1"], preserve_partity=True, name="pipi_resonance_1"),
            ],
        # This is the J/Psi meson. It is defined as a resonance, but we assue a constant lineshape, since it is a extremely narrow resonance. 
        # By setting this resonance to the helicity scheme, the resulting function will expect helicity couplings and not ls couplings
        # Note, that the preserve_partity flag is still set to True, but does not have any effect in the helicity scheme, as here parity conservation can not be enforced by truncating the couplings
        # For the J/Psi decay, we know, that the helicity couplings are all the same and we can thus just fix them in the fitter.
        (3, 4): [Resonance(Node((3, 4)), quantum_numbers=QN(2, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=True, name="J/Psi_1", scheme="helicity")],

        # This is the decaying B0 meson. It is defined as a resonance, but since this is a decay amplitude, the description is not important. Only the QN have to be correct aswell as the parity conservation, since this controlls the automated ls couplings
        0: [Resonance(Node(0), quantum_numbers=QN(0, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="B0_1")]
    }

    m_pip_JPsi = topology_pip_JPsi.nodes[(1,3,4)].mass(momenta=momenta)
    resonances_pip_JPsi = {
        (1, 3, 4): [
            # Here the pip JPsi resonances go
            # These will decay strong, so we need to conserve parity
            Resonance(Node((1, 3, 4)), quantum_numbers=QN(2, -1), lineshape=BW_lineshape(m_pip_JPsi), argnames=["mass_resonance_2", "width_resonance_2"], preserve_partity=True, name="J/Psi_resonance_1"),
            ],
        # This is the J/Psi meson. It is defined as a resonance, but we assue a constant lineshape, since it is a extremely narrow resonance. 
        # By setting this resonance to the helicity scheme, the resulting function will expect helicity couplings and not ls couplings
        # Note, that the preserve_partity flag is still set to True, but does not have any effect in the helicity scheme, as here parity conservation can not be enforced by truncating the couplings
        # For the J/Psi decay, we know, that the helicity couplings are all the same and we can thus just fix them in the fitter.
        (3, 4): [Resonance(Node((3, 4)), quantum_numbers=QN(2, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=True, name="J/Psi_2", scheme="helicity")],

        # This is the decaying B0 meson. It is defined as a resonance, but since this is a decay amplitude, the description is not important. Only the QN have to be correct aswell as the parity conservation, since this controlls the automated ls couplings
        0: [Resonance(Node(0), quantum_numbers=QN(0, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="B0_2")]
    }

    m_pi_m_JPsi = topology_pi_m_JPsi.nodes[(2,3,4)].mass(momenta=momenta)
    resonances_pi_m_JPsi = {
        (2, 3, 4): [
            # Here the pi- JPsi resonances go
            # These will decay strong, so we need to conserve parity
            Resonance(Node((2, 3, 4)), quantum_numbers=QN(0, -1), lineshape=BW_lineshape(m_pi_m_JPsi), argnames=["mass_resonance_3", "width_resonance_3"], preserve_partity=True, name="J/Psi_resonance_2"),
            ],
        # This is the J/Psi meson. It is defined as a resonance, but we assue a constant lineshape, since it is a extremely narrow resonance. 
        # By setting this resonance to the helicity scheme, the resulting function will expect helicity couplings and not ls couplings
        # Note, that the preserve_partity flag is still set to True, but does not have any effect in the helicity scheme, as here parity conservation can not be enforced by truncating the couplings
        # For the J/Psi decay, we know, that the helicity couplings are all the same and we can thus just fix them in the fitter.
        (3, 4): [Resonance(Node((3, 4)), quantum_numbers=QN(2, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=True, name="J/Psi_3", scheme="helicity")],

        # This is the decaying B0 meson. It is defined as a resonance, but since this is a decay amplitude, the description is not important. Only the QN have to be correct aswell as the parity conservation, since this controlls the automated ls couplings
        0: [Resonance(Node(0), quantum_numbers=QN(0, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="B0_3")]
    }

    return resonances_pi_pi, resonances_pip_JPsi, resonances_pi_m_JPsi


def phasespace_momenta():
    import phasespace
    B0_MASS = 5.27963
    PION_MASS = 0.13957018
    MU_MASS = 0.1056583715

    weights, particles = phasespace.nbody_decay(B0_MASS,
                                            [PION_MASS, PION_MASS, MU_MASS, MU_MASS]).generate(n_events=10)
    momenta = {
        1: np.array(particles["p_0"]),
        2: np.array(particles["p_1"]),
        3: np.array(particles["p_2"]),
        4: np.array(particles["p_3"]),
    }
    return momenta

def shortFourBodyAmplitudeBW():
    final_state_qn = {
        1: QN(0, 1), # pi+
        2: QN(0, -1), # pi-
        3: QN(1, 1), # mu+
        4: QN(1, -1) # mu-
    }
    momenta = phasespace_momenta()

    resonances_pi_pi, resonances_pip_JPsi, resonances_pi_m_JPsi = resonances(momenta)
    chain1 = MultiChain(
        topology = topology_pipi,
        resonances = resonances_pi_pi,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    chain2 = MultiChain(
        topology = topology_pip_JPsi,
        resonances = resonances_pip_JPsi,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    chain3 = MultiChain(
        topology = topology_pi_m_JPsi,
        resonances = resonances_pi_m_JPsi,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    # For semileptonics no actual alignment is needed, as we are only interested in one single decay chain
    # But doing it this way allows to use the convenience functions of the combiner
    full = ChainCombiner([chain1, chain2, chain3])

    # The unpolarized amplitude is the simplest one, and the default case in LHCb
    unpolarized, argnames = full.unpolarized_amplitude(
        full.generate_couplings() # This is a helper function to generate the couplings for the hadronic system, if you want to restrict them, you will have to do it manually.
                                    # Alternatively you can also restrict the couplings in the fitter later.      
        )
    # argnames are the names of the arguments of the function, which are the masses and widths of the resonances and the couplings
    # The order of argnames is the order of the arguments in the function
    # The function can be called with positional arguments, or with keyword arguments
    # so unpolarized(*[1, 2, 3, 4, 5, 6]) is the same as unpolarized(mass_resonance_1=1, width_resonance_1=2, mass_resonance_2=3, width_resonance_2=4, mass_resonance_3=5, width_resonance_3=6), omitting the couplings
    print(argnames)

    # an issue with jax, where the internal caching structure needs to be prewarmed, so that in the compilation step everything, that can be static, is static
    print(unpolarized(*([1] * len(argnames))))
    
    # we can now jit the function
    unpolarized = jit(unpolarized) 
    print(unpolarized(*([1] * len(argnames))))


    # for the gradient calculation we need to define a log likelihood function or something, that produces a single value
    def LL(*args):
        # Warning: This is not the correct log likelihood function, but just a dummy
        # The amplitudes do not come normalized, so this can not be used for fitting directly
        return np.sum(
            np.log(unpolarized(*args))
                )
    # we can calc the gradient of the log likelihood function
    unpolarized_grad = jit(grad(LL, argnums=[i for i in range(len(argnames))]))

    # and a test call
    print(unpolarized_grad(*([1.0] * len(argnames))))


    # Other options for amplitudes, one might be interested int
    # polarized, lambdas ,polarized_argnames = full.polarized_amplitude(full.generate_couplings())
    # print(lambdas)
    # lambda_values = [0, 0, 0, 1, 1]
    # print(polarized(*lambda_values,*([1] * len(polarized_argnames))) )

    # matrix_function, matrix_argnames = full.matrix_function(full.generate_couplings())
    # print(matrix_argnames)
    # print(matrix_function(0, *([1] * len(argnames))) )


if __name__ == "__main__":
    shortFourBodyAmplitudeBW()