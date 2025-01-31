from utils import make_four_vectors, constant_lineshape
from decayamplitude.resonance import Resonance
from decayamplitude.rotation import QN, Angular
from decayamplitude.chain import DecayChain
from decayamplitude.combiner import ChainCombiner

import numpy as np



from decayangle.decay_topology import Topology, Node
from decayangle.config import config as decayangle_config

# we want to define our chains by hand, so we need to turn off the automatic sorting
decayangle_config.sorting = "off" 

def resonances() -> tuple[dict]:
    resonances1 = {
        (2,3): Resonance(Node((2, 3)), 0, -1, lineshape=constant_lineshape, argnames=[]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[])
    }

    resonances2 = {
        (1, 2): Resonance(Node((1, 2)), 3, -1, lineshape=constant_lineshape, argnames=[]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[])
    }

    resonances3 = {
        (1, 2): Resonance(Node((1, 2)), 1, -1, lineshape=constant_lineshape, argnames=[]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[])
    }

    resonances_dpd = {
        (2,3): Resonance(Node((2, 3)), 4, -1, lineshape=constant_lineshape, argnames=[]),
        0: Resonance(Node(0), 1, 1, lineshape=constant_lineshape, argnames=[])
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
            "ls_couplings":{
                (2, 2) : 1
            }
        }, 
        resonances1[0].id : {
            "ls_couplings":{
                (2, 1) : 1
            }
        }
    }
    arguments2 = {
        resonances2[(1,2)].id : {
            "ls_couplings":{
                (2, 3) : 1
            }
        }, 
        resonances2[0].id : {
            "ls_couplings":{
                (2, 3) : 1
            }
        }
    }

    arguments3 = {
        resonances3[(1,2)].id : {
            "ls_couplings":{
                (2, 3) : 1
            }
        }, 
        resonances3[0].id : {
            "ls_couplings":{
                (2, 1) : 1
            }
        }
    }

    arguments_dpd = {
        resonances_dpd[(2,3)].id : {
            "ls_couplings": {
                (4, 2): 1
            },
        },
        resonances_dpd[0].id : {
            "ls_couplings": {
                (4, 3): 1
            }
        }
    }

    arguments = {}
    arguments.update(arguments1)
    arguments.update(arguments2)
    arguments.update(arguments3)
    arguments.update(arguments_dpd)

    helicity_angles = topology1.helicity_angles(momenta)
    helicity_angles.update(topology2.helicity_angles(momenta))


    full = ChainCombiner([decay, decay2, decay3, decay_dpd])
    print(full.combined_function(0, {1:1, 2:0, 3:0}, helicity_angles ,arguments))

    all_helicities = Angular.generate_helicities(*[final_state_qn[key].angular for key in final_state_qn.keys()])
    all_helicities = [
        {key: helicity[i] for i, key in enumerate(final_state_qn.keys())}
        for helicity in all_helicities
    ]

    full_matrix_1 = full.combined_matrix(-1, helicity_angles, arguments)
    full_matrix_2 = full.combined_matrix(1, helicity_angles, arguments)
    print(sum(abs(v)**2 for v in full_matrix_1.values()) + 
          sum(abs(v)**2 for v in full_matrix_2.values()))

    # for key, vlaue in full_matrix.items():
    #     print(key, abs(vlaue))

if __name__ == "__main__":
    threeBodyAmplitude()