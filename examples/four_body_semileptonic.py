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
decayangle_config.sorting = "off"

from jax import jit, grad

from collections import defaultdict

# Since we are semi leptonic there is only one important decay topology
# The (1, 2) system is the hadronic system, which is the one we are interested in
# Decay definition: B0 -> D0 h mu nu  or in the notation of the library 0 -> ((1, 2), (3, 4))
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
            Resonance(Node((1, 2)), quantum_numbers=QN(0, 1), lineshape=BW_lineshape(m_12), argnames=["D_2300_M", "D_2300_Gamma"], preserve_partity=True, name="D*0(2300)"),
            Resonance(Node((1, 2)), quantum_numbers=QN(4, 1), lineshape=BW_lineshape(m_12), argnames=["D_2460_M", "D_2460_Gamma"], preserve_partity=True, name="D*2(2460)"),
            Resonance(Node((1, 2)), quantum_numbers=QN(2, -1), lineshape=BW_lineshape(m_12), argnames=["D_2600_M", "D_2600_Gamma"], preserve_partity=True, name="D*1(2600)"),
            Resonance(Node((1, 2)), quantum_numbers=QN(0, 1), lineshape=constant_lineshape, argnames=[], preserve_partity=True, name="Non-Resonant-S-Wave"),
            # Resonance(Node((1, 2)), quantum_numbers=QN(J, P), lineshape=BW_lineshape(m_12), argnames=["mass_resonance_n", "width_resonance_n"], preserve_partity=True), # template for further resonances
            ],
        # This is the W boson. It is defined as a resonance, but we assue a constant lineshape in this mass regime. One could use a more complicated one aswell.
        (3, 4): [Resonance(Node((3, 4)), quantum_numbers=QN(2, -1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="W")],

        # This is the decaying B0 meson. It is defined as a resonance, but since this is a decay amplitude, the description is not important. Only the QN have to be correct. 
        0: [Resonance(Node(0), quantum_numbers=QN(0, 1), lineshape=constant_lineshape, argnames=[], preserve_partity=False, name="B0")],
    }
    return resonances_hadronic


def phasespace_momenta():
    # dependence on the phasespace library, dont use, if you dont have it installed
    # Installing via pip may destroy you existing  cuda setup
    # This is a simple example, where we use the phasespace library to generate the momenta
    import phasespace
    B0_MASS = 5.27963e3
    PION_MASS = 0.13957018e3
    D0_MASS = 1.86483e3
    MU_MASS = 0.1056583715e3
    NU_MASS = 0
    weights, particles = phasespace.nbody_decay(B0_MASS,
                                            [D0_MASS, PION_MASS, MU_MASS, NU_MASS]).generate(n_events=100000)
    momenta = {
        1: np.array(particles["p_0"]),
        2: np.array(particles["p_1"]),
        3: np.array(particles["p_2"]),
        4: np.array(particles["p_3"]),
    }
    return momenta

def shortFourBodyAmplitudeBW():
    final_state_qn = {
        1: QN(0, 1), # D0
        2: QN(0, 1), # h
        3: QN(1, 1), # mu
        4: QN(1, -1) # nu
    }
    momenta = phasespace_momenta()

    resonances_hadronic = resonances_BW(momenta)
    chain1 = MultiChain(
        topology = topology1,
        resonances = resonances_hadronic,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    # For semileptonics no actual alignment is needed, as we are only interested in one single decay chain
    # But doing it this way allows to use the convenience functions of the combiner. For a one chais setup the combiner will not align anyways.
    full = ChainCombiner([chain1])

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

    # an issue with jax, where the internal caching structure needs to be prewarmed, so that in the compilation step the correct types are inferred
    print(unpolarized(*([1.0] * len(argnames))))
    
    # we can now jit the function, to make it faster after the compile
    unpolarized = jit(unpolarized) 
    print(unpolarized(*([1.0] * len(argnames))))


    # for the gradient calculation we need to define a log likelihood function or something, that produces a single value
    def LL(*args):
        return np.sum(
            np.log(unpolarized(*args))
                )
    # we can calc the gradient of the log likelihood function
    unpolarized_grad = jit(grad(LL, argnums=[i for i in range(len(argnames))]))

    # and a test call (may take quite some time)
    # print(unpolarized_grad(*([1.0] * len(argnames))))

    # Further calls will be much faster, since we dont need to compile again
    # print(unpolarized_grad(*([1.0] * len(argnames))))
    # print(unpolarized_grad(*([2.0] * len(argnames))))


    # Other options for amplitudes, one might be interested in
    # polarized, lambdas ,polarized_argnames = full.polarized_amplitude(full.generate_couplings())
    # print(lambdas)
    # lambda_values = [0, 0, 0, 1, 1]
    # print(polarized(*lambda_values,*([1] * len(polarized_argnames))) )

    # matrix_function, matrix_argnames = full.matrix_function(full.generate_couplings())
    # print(matrix_argnames)
    # print(matrix_function(0, *([1] * len(argnames))) )


    # # quick plot to show the resonant structure can be confiremd visually
    # import matplotlib.pyplot as plt
    # # set couplings to 1.0
    # param_dict = {param_name: 1.0 for param_name in argnames}
    # param_dict.update({
    #     # made up masses and widths
    #     'D_2300_Gamma': 20, 
    #     'D_2460_Gamma': 60, 
    #     'D_2460_M': 2460, 
    #     'D_2300_M': 2300, 
    #     'D_2600_M': 2600, 
    #     'D_2600_Gamma': 80,
    #     # set the non resonant part to a low value, since otherwise it will dominate the plot
    #     "Non_Resonant_S_Wave_to_particle_1_particle_2_LS_0_0_real": 0.00001,
    #     "Non_Resonant_S_Wave_to_particle_1_particle_2_LS_0_0_imaginary": 0.0,
    # })
    # del param_dict['Non_Resonant_S_Wave_to_particle_1_particle_2_LS_0_0_real']
    # import numpy as onp # original numpy plays nicer with matplotlib, so we convert the results to numpy
    # weights = onp.array(unpolarized(**param_dict))
    # hadronic_mass = topology1.nodes[(1,2)].mass(momenta=momenta)
    # plt.hist(onp.array(hadronic_mass), bins=100, weights=weights, label="Hadronic mass")
    # plt.show()



if __name__ == "__main__":
    shortFourBodyAmplitudeBW()