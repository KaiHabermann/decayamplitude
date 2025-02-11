from utils import make_four_vectors, constant_lineshape, BW_lineshape
from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN, Angular
from decayamplitude.chain import DecayChain, MultiChain
from decayamplitude.combiner import ChainCombiner

from decayamplitude.backend import numpy as np



from decayangle.decay_topology import Topology, Node
from decayangle.kinematics import mass
from decayangle.config import config as decayangle_config

from collections import defaultdict

# we want to define our chains by hand, so we need to turn off the automatic sorting
decayangle_config.sorting = "off" 

def resonances() -> tuple[dict]:
    resonances1 = {
        (2,3): Resonance(Node((2, 3)), quantum_numbers=QN(0, -1), lineshape=constant_lineshape, argnames=[]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[], preserve_partity=False)
    }

    resonances2 = {
        (1, 2): Resonance(Node((1, 2)), quantum_numbers=QN(3, -1), lineshape=constant_lineshape, argnames=[]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[], preserve_partity=False)
    }

    resonances3 = {
        (1, 2): Resonance(Node((1, 2)), quantum_numbers=QN(1, -1), lineshape=constant_lineshape, argnames=[]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[], preserve_partity=False)
    }

    resonances_dpd = {
        (2,3): Resonance(Node((2, 3)), quantum_numbers=QN(4, -1), lineshape=constant_lineshape, argnames=[]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[], preserve_partity=False)
    }

    return resonances1, resonances2, resonances3, resonances_dpd

def resonances_BW(momenta) -> tuple[dict]:
    nodes = Topology(0, decay_topology=((2,3), 1)).nodes
    nodes_2 = Topology(0, decay_topology=((1,2), 3)).nodes

    resonances1 = {
        (2,3): Resonance(nodes[(2, 3)], quantum_numbers=QN(0, -1), lineshape=BW_lineshape(nodes[(2, 3)].mass(momenta)), argnames=["gamma1", "m01"]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[], preserve_partity=False)
    }

    resonances2 = {
        (1, 2): Resonance(nodes_2[(1, 2)], quantum_numbers=QN(3, -1), lineshape=BW_lineshape(nodes_2[(1, 2)].mass(momenta)), argnames=["gamma2", "m01"]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[], preserve_partity=False)
    }

    resonances3 = {
        (1, 2): Resonance(Node((1, 2)), quantum_numbers=QN(1, -1), lineshape=BW_lineshape(nodes_2[(1, 2)].mass(momenta)), argnames=["gamma1", "m01"]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[], preserve_partity=False)
    }

    resonances_dpd = {
        (2,3): Resonance(nodes[(2, 3)], quantum_numbers=QN(4, -1), lineshape=BW_lineshape(nodes[(2, 3)].mass(momenta)), argnames=["gamma1", "m01"]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[], preserve_partity=False)
    }

    return resonances1, resonances2, resonances3, resonances_dpd

def threeBodyAmplitude():
    """
    Whe create an amplitdue for a three-body decay chain with the following topology:
    0 -> ((2,3)-> 2 3) 1
    
    """
    topology1 = Topology(
        0,
        decay_topology=((2,3), 1)
    )

    topology2 = Topology(
        0,
        decay_topology=((1, 2), 3)
    )

    # we need to define the momenta for the decay chain
    momenta = make_four_vectors(1,2,np.linspace(0,np.pi,10))

    final_state_qn = {
            1: QN(1, 1),
            2: QN(2, 1),
            3: QN(0, 1)
        }
    resonances1, resonances2, resonances3, resonances_dpd = resonances()
    decay = DecayChain(
        topology = topology1,
        resonances = resonances1,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    decay2 = DecayChain(
        topology = topology2,
        resonances = resonances2,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    decay3 = DecayChain(
        topology = topology2,
        resonances = resonances3,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    decay_dpd = DecayChain(
        topology = topology1,
        resonances = resonances_dpd,
        momenta = momenta,
        final_state_qn = final_state_qn
    )

    arguments1 = {
        resonances1[(2,3)].id : {
            "couplings":{
                (2, 2) : 1
            }
        }, 
        resonances1[0].id : {
            "couplings":{
                (2, 1) : 1
            }
        }
    }
    arguments2 = {
        resonances2[(1,2)].id : {
            "couplings":{
                (2, 3) : 1
            }
        }, 
        resonances2[0].id : {
            "couplings":{
                (2, 3) : 1
            }
        }
    }

    arguments3 = {
        resonances3[(1,2)].id : {
            "couplings":{
                (2, 3) : 1
            }
        }, 
        resonances3[0].id : {
            "couplings":{
                (2, 1) : 1
            }
        }
    }

    arguments_dpd = {
        resonances_dpd[(2,3)].id : {
            "couplings": {
                (4, 2): 1
            },
        },
        resonances_dpd[0].id : {
            "couplings": {
                (4, 3): 1
            }
        }
    }

    arguments = {}
    arguments.update(arguments1)
    arguments.update(arguments2)
    arguments.update(arguments3)
    arguments.update(arguments_dpd)


    full = ChainCombiner([decay, decay2, decay3, decay_dpd])
    print(full.combined_function(0, {1:1, 2:0, 3:0} ,arguments))

    all_helicities = Angular.generate_helicities(*[final_state_qn[key].angular for key in final_state_qn.keys()])
    all_helicities = [
        {key: helicity[i] for i, key in enumerate(final_state_qn.keys())}
        for helicity in all_helicities
    ]

    full_matrix_1 = full.combined_matrix(-1, arguments)
    full_matrix_2 = full.combined_matrix(1, arguments)
    print(sum(abs(v)**2 for v in full_matrix_1.values()) + 
          sum(abs(v)**2 for v in full_matrix_2.values()))
    

    # Advantage of this setup is, that we only have one alignment step per topology
    ch1 = MultiChain.from_chains([decay, decay_dpd])
    ch2 = MultiChain.from_chains([decay2, decay3])
    full_multi = ChainCombiner([ch1, ch2])
    full_matrix_multi_1 = full_multi.combined_matrix(-1, arguments)
    full_matrix_multi_2 = full_multi.combined_matrix(1, arguments)
    print(sum(abs(v)**2 for v in full_matrix_multi_1.values()) + 
          sum(abs(v)**2 for v in full_matrix_multi_2.values())
    )


def shortThreeBodyAmplitude():
    """
    We can also combine the chains in a shorter way, by using the MultiChain class
    """
    final_state_qn = {
            1: QN(1, 1),
            2: QN(2, 1),
            3: QN(0, 1)
        }
    resonances1, resonances2, resonances3, resonances_dpd = resonances()
    momenta = make_four_vectors(1,2,np.linspace(0,np.pi,10))
    topology1 = Topology(
        0,
        decay_topology=((2,3), 1)
    )
    topology2 = Topology(
        0,
        decay_topology=((1, 2), 3)
    )

    chain1 = MultiChain.from_chains([
        DecayChain(
            topology = topology1,
            resonances = resonances1,
            momenta = momenta,
            final_state_qn = final_state_qn
        ),
        DecayChain(
            topology = topology1,
            resonances = resonances_dpd,
            momenta = momenta,
            final_state_qn = final_state_qn
        )
    ])

    chain2 = MultiChain.from_chains([
        DecayChain(
            topology = topology2,
            resonances = resonances2,
            momenta = momenta,
            final_state_qn = final_state_qn,
        ),
        DecayChain(
            topology = topology2,
            resonances = resonances3,
            momenta = momenta,
            final_state_qn = final_state_qn
        )
    ])

    merged_resonances = defaultdict(list)
    for key, resonance in resonances3.items():
        merged_resonances[key].append(resonance)
    for key, resonance in resonances2.items():
        merged_resonances[key].append(resonance)
    merged_resonances = dict(merged_resonances)

    merged_resonances[0] = [merged_resonances[0][0]]

    full = ChainCombiner([chain1, chain2])
    arguments = full.generate_couplings()

    matrix1 = full.combined_matrix(-1, arguments)
    matrix2 = full.combined_matrix(1, arguments)
    print(sum(abs(v)**2 for v in matrix1.values()) + 
          sum(abs(v)**2 for v in matrix2.values())
    )
    unpolarized, argnames = full.unpolarized_amplitude(full.generate_couplings())
    print(argnames)
    print(unpolarized(*([1] * len(argnames))))
    full2 = ChainCombiner([chain1, MultiChain(topology2, momenta=momenta, resonances=merged_resonances, final_state_qn=final_state_qn)])
    unpolarized, argnames = full2.unpolarized_amplitude(full2.generate_couplings())
    print(argnames)
    print(unpolarized(*([1] * len(argnames))))


def shortThreeBodyAmplitudeBW():
    final_state_qn = {
        1: QN(1, 1),
        2: QN(2, 1),
        3: QN(0, 1)
    }
    momenta = make_four_vectors(1,2,np.linspace(0,np.pi,10))
    topology1 = Topology(
        0,
        decay_topology=((2,3), 1)
    )
    topology2 = Topology(
        0,
        decay_topology=((1, 2), 3)
    )
    resonances1, resonances2, resonances3, resonances_dpd = resonances_BW(momenta)
    chain1 = MultiChain.from_chains([
        DecayChain(
            topology = topology1,
            resonances = resonances1,
            momenta = momenta,
            final_state_qn = final_state_qn
        ),
        DecayChain(
            topology = topology1,
            resonances = resonances_dpd,
            momenta = momenta,
            final_state_qn = final_state_qn
        )
    ])

    chain2 = MultiChain.from_chains([
        DecayChain(
            topology = topology2,
            resonances = resonances2,
            momenta = momenta,
            final_state_qn = final_state_qn,
        ),
        DecayChain(
            topology = topology2,
            resonances = resonances3,
            momenta = momenta,
            final_state_qn = final_state_qn
        )
    ])

    full = ChainCombiner([chain1, chain2])
    unpolarized, argnames = full.unpolarized_amplitude(full.generate_couplings())
    print(argnames)
    print(unpolarized(*([1] * len(argnames))) )

    polarized, lambdas ,polarized_argnames = full.polarized_amplitude(full.generate_couplings())
    print(lambdas)
    lambda_values = [1, 1, 0, 0]
    print(polarized(*lambda_values,*([1] * len(polarized_argnames))) )

    matrx_function, matrix_argnames = full.matrix_function(full.generate_couplings())
    print(matrix_argnames)
    print(matrx_function(1, *([1] * len(argnames))) )


if __name__ == "__main__":
    threeBodyAmplitude()
    shortThreeBodyAmplitude()
    shortThreeBodyAmplitudeBW()